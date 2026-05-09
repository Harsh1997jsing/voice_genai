from pinecone import Pinecone
from openai import OpenAI
from openai import AsyncOpenAI
import uuid
import time
import asyncio
import threading
import inspect
from app.models.kb_model import ExtractionResult
from app.core.config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

index_host = pc.describe_index(
    settings.PINECONE_INDEX_NAME
).host

async_index = pc.IndexAsyncio(host=index_host)
index = pc.Index(settings.PINECONE_INDEX_NAME)
# async_index = pc.IndexAsyncio(settings.PINECONE_INDEX_NAME)
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
openai_async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


EMBED_CACHE_TTL_SECS = 900
SEARCH_CACHE_TTL_SECS = 120


_search_cache: dict = {}
_embed_cache: dict = {}
_cache_lock = threading.Lock()

def _cache_get(cache: dict, key):
    with _cache_lock:
        entry = cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del cache[key]
            return None
        return value

def _cache_set(cache: dict, key, value, ttl_secs: int):
    with _cache_lock:
        cache[key] = (time.monotonic() + ttl_secs, value)


def _normalize_query(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def get_embedding(text: str):
    cached = _cache_get(_embed_cache, text)
    # print(f"Embedding Cache - Key: '{text}' | Hit: {cached is not None}")
    if cached is not None:
        return cached

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    print(f"Generated new embedding for text: '{text[:30]}...' | Length: {len(embedding)}")
    _cache_set(_embed_cache, text, embedding, EMBED_CACHE_TTL_SECS)
    return embedding

async def get_embedding_async(text: str):
    key = _normalize_query(text) 
    cached = _cache_get(_embed_cache, key)
    if cached is not None:
        return cached
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        # dimensions=512  # ← also reduce dims (see below)
    )
    embedding = response.data[0].embedding
    _cache_set(_embed_cache, key, embedding, EMBED_CACHE_TTL_SECS)
    return embedding


def add_to_kb(result: ExtractionResult, user_id: int) -> dict:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
    )

    chunks = splitter.split_text(result.text)

    if not chunks:
        raise ValueError(f"No chunks from '{result.metadata.filename}'")

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embedding,
            "metadata": {
                "text": chunk,
                "chunk_index": i,
                "filename": result.metadata.filename,
                "has_tables": result.metadata.has_tables,
                "word_count": len(chunk.split()),
            }
        })

    # Batch upsert (max 100 per call)
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100], namespace=f"{user_id}_kb")

    return {
        "filename": result.metadata.filename,
        "chunks_created": len(chunks),
    }



async def _query_namespace(query_vector, top_k: int, namespace: str):
    results = await async_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    # print(f"Raw KB Search Results [{namespace}]: {results}")
    return [
        match["metadata"]["text"]
        for match in results["matches"]
    ]


async def search_kb(query: str, top_k: int = 3, user_id: int = None):
    cache_key = (query, top_k, user_id)
    cached = _cache_get(_search_cache, cache_key)
    if cached is not None:
        if inspect.isawaitable(cached):
            cached = await cached
            _cache_set(_search_cache, cache_key, cached, SEARCH_CACHE_TTL_SECS)
        return cached

    query_vector = await get_embedding_async(query)
    primary_namespace = f"{user_id}_kb"
    # fallback_namespace = "default_user_kb"

    matches = await _query_namespace(query_vector, top_k, primary_namespace)
    if inspect.isawaitable(matches):
        matches = await matches
    # if not matches and primary_namespace != fallback_namespace:
    #     print(f"No results in {primary_namespace}, trying {fallback_namespace}")
    #     matches = _query_namespace(query_vector, top_k, fallback_namespace)

    _cache_set(_search_cache, cache_key, matches, SEARCH_CACHE_TTL_SECS)
    return matches


async def search_kb_async(query: str, top_k: int = 3, user_id: int = 1):
    return await search_kb(query, top_k, user_id)


async def generate_answer_async(query: str, user_id: int = 1):
    docs = await search_kb_async(query, user_id=user_id)

    context = "\n".join(docs)

    response = await openai_async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on context"},
            {"role": "user", "content": f"{context}\n\nQ: {query}"}
        ]
    )

    return response.choices[0].message.content
