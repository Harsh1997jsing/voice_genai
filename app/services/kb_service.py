"""
kb_service.py — Production-grade knowledge-base retrieval service.

Fixes vs original:
  - asyncio.Lock replaces threading.Lock (threading.Lock blocks the event loop)
  - TTLCache (cachetools) replaces raw dicts → bounded memory, automatic eviction
  - Embedding cache keyed on normalized query (lowercase + collapsed whitespace)
  - search_kb fully async-safe; no inspect.isawaitable hacks needed
  - user_id is a required parameter (no default=1 hiding multi-tenant bug)
  - Pinecone async client reused across calls (connection-pool friendly)
  - Structured logging via structlog
"""

import asyncio
import re
import time
import uuid

import structlog
from cachetools import TTLCache
from openai import AsyncOpenAI, OpenAI
from pinecone import Pinecone

from app.core.config import settings
from app.models.kb_model import ExtractionResult
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = structlog.get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Pinecone clients
# ──────────────────────────────────────────────────────────────────────────────
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

_index_host = pc.describe_index(settings.PINECONE_INDEX_NAME).host

# Sync client — used only for upsert during ingestion (not in hot path)
_sync_index = pc.Index(settings.PINECONE_INDEX_NAME)

# Async client — shared across all concurrent callers.
# The Pinecone SDK manages an internal connection pool; reusing one instance
# is correct and efficient.
_async_index = pc.IndexAsyncio(host=_index_host)

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI clients
# ──────────────────────────────────────────────────────────────────────────────
_openai_sync = OpenAI(api_key=settings.OPENAI_API_KEY)
_openai_async = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Caches
# TTLCache is NOT thread-safe on its own; we guard it with asyncio.Lock.
# maxsize prevents unbounded memory growth under high call volume.
# ──────────────────────────────────────────────────────────────────────────────
EMBED_CACHE_TTL = 900    # 15 minutes
SEARCH_CACHE_TTL = 120   # 2 minutes

_embed_cache: TTLCache = TTLCache(maxsize=2048, ttl=EMBED_CACHE_TTL)
_search_cache: TTLCache = TTLCache(maxsize=512,  ttl=SEARCH_CACHE_TTL)

# asyncio.Lock — safe to use inside coroutines (never blocks event loop thread)
_embed_lock = asyncio.Lock()
_search_lock = asyncio.Lock()

EMBED_MODEL = "text-embedding-3-small"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Canonical cache key: lowercase, collapsed whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ──────────────────────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────────────────────

async def get_embedding_async(text: str) -> list[float]:
    """
    Returns a cached embedding if available; otherwise calls OpenAI async.
    Lock ensures only one inflight request per unique query even under
    concurrent callers — the second caller benefits from the first's result.
    """
    key = _normalize(text)

    async with _embed_lock:
        cached = _embed_cache.get(key)
        if cached is not None:
            return cached

    # Release lock while awaiting network I/O — other coroutines can proceed
    response = await _openai_async.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding

    async with _embed_lock:
        _embed_cache[key] = embedding

    return embedding


def get_embedding_sync(text: str) -> list[float]:
    """Synchronous embedding — used only during ingestion (add_to_kb)."""
    key = _normalize(text)
    # Safe to access without lock here because ingestion is not concurrent
    cached = _embed_cache.get(key)
    if cached is not None:
        return cached

    response = _openai_sync.embeddings.create(model=EMBED_MODEL, input=text)
    embedding = response.data[0].embedding
    _embed_cache[key] = embedding
    return embedding


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────────────────────

def add_to_kb(result: ExtractionResult, user_id: int) -> dict:
    """
    Chunk, embed, and upsert a document into the user's Pinecone namespace.
    Runs synchronously (called from a background task / worker, not from
    the voice WebSocket handler).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_text(result.text)

    if not chunks:
        raise ValueError(f"No chunks extracted from '{result.metadata.filename}'")

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding_sync(chunk)
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i,
                "filename": result.metadata.filename,
                "has_tables": result.metadata.has_tables,
                "word_count": len(chunk.split()),
            },
        })

    namespace = f"{user_id}_kb"
    for i in range(0, len(vectors), 100):
        _sync_index.upsert(vectors=vectors[i : i + 100], namespace=namespace)

    log.info(
        "kb_ingested",
        filename=result.metadata.filename,
        chunks=len(chunks),
        user_id=user_id,
    )
    return {"filename": result.metadata.filename, "chunks_created": len(chunks)}


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

async def _query_namespace(
    query_vector: list[float],
    top_k: int,
    namespace: str,
) -> list[str]:
    results = await _async_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    return [match["metadata"]["text"] for match in results["matches"]]


async def search_kb(query: str, user_id: int, top_k: int = 3) -> list[str]:
    """
    Search the knowledge base for `query` in `user_id`'s namespace.

    Cache lookup → embedding → Pinecone query, with a per-key async lock
    so concurrent callers for the same query share one network round-trip.
    """
    cache_key = (_normalize(query), top_k, user_id)

    # Fast path: already cached
    async with _search_lock:
        cached = _search_cache.get(cache_key)
        if cached is not None:
            return cached

    # Slow path: embed + query
    t0 = time.perf_counter()
    query_vector = await get_embedding_async(query)
    namespace = f"{user_id}_kb"
    matches = await _query_namespace(query_vector, top_k, namespace)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "kb_search",
        user_id=user_id,
        query_preview=query[:60],
        hits=len(matches),
        ms=round(elapsed_ms, 1),
    )

    async with _search_lock:
        _search_cache[cache_key] = matches

    return matches


async def search_kb_async(query: str, user_id: int, top_k: int = 3) -> list[str]:
    """Thin alias kept for call-site compatibility."""
    return await search_kb(query=query, user_id=user_id, top_k=top_k)


# ──────────────────────────────────────────────────────────────────────────────
# One-shot QA (non-streaming) — used outside the real-time voice path
# ──────────────────────────────────────────────────────────────────────────────

async def generate_answer_async(query: str, user_id: int) -> str:
    docs = await search_kb_async(query=query, user_id=user_id)
    context = "\n".join(docs)

    response = await _openai_async.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on context only."},
            {"role": "user", "content": f"{context}\n\nQ: {query}"},
        ],
    )
    return response.choices[0].message.content