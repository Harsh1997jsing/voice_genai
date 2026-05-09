import time
from app.services.kb_service import search_kb_async
from app.services.llm_service import stream_llm


# Fallback response when query is out of scope
# Keep it short — this is a phone call
FALLBACK_RESPONSE = (
    "I can only help with questions related to our products and services. "
    "Is there anything specific about that I can help you with?"
)


async def rag_stream(query: str, user_id: int, trace_id: str = "na"):
    # Latency optimization:
    # Skip classifier round-trip in live call path to reduce first-token delay.
    t0 = time.perf_counter()
    print(f"[RAG] Starting RAG stream for query: '{query}'")
    docs = await search_kb_async(query=query, user_id=user_id)
    retrieve_ms = (time.perf_counter() - t0) * 1000
    print(f"[LATENCY] trace={trace_id} stage=rag_retrieve ms={retrieve_ms:.1f}")

    if not docs:
        yield "I don't have specific information about that, but I'm happy to help with anything else related to our services."
        return

    context = "\n".join(docs)
    print(f"[RAG] Retrieved {len(docs)} docs for query: '{query}'")

    async for chunk in stream_llm(query, context, trace_id=trace_id):
        yield chunk
