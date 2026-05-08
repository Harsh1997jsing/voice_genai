from openai import AsyncOpenAI
from app.core.config import settings
import time

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def stream_llm(query: str, context: str, trace_id: str = "na"):
    prompt = f"""
    Answer ONLY using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    first_token_logged = False
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            if not first_token_logged:
                first_token_ms = (time.perf_counter() - t0) * 1000
                print(f"[LATENCY] trace={trace_id} stage=llm_first_token ms={first_token_ms:.1f}")
                first_token_logged = True
            yield chunk.choices[0].delta.content


async def classify_query(query: str) -> bool:
    """
    Returns True if query is in scope OR is a greeting/conversational filler.
    Returns False only for clearly unrelated topics.
    """
    from app.core.config import settings
    from openai import AsyncOpenAI
 
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
 
    system_prompt = """You are a query classifier for a health insurance voice AI agent.
 
Classify the user query into one of these responses:
- "yes" → query is about health insurance, plans, claims, hospitals, renewals, payments, policy, or general greetings/small talk like "hello", "hi", "how are you", "okay", "thank you"
- "no" → query is clearly about something completely unrelated to health insurance (weather, cricket, cooking, travel, etc.)
 
IMPORTANT RULES:
1. Greetings and conversational phrases (hello, hi, okay, yes, no, thank you) → always "yes"
2. Queries mixing any Indian language (Hindi, Tamil, Telugu, Kannada, Malayalam, Odia, Punjabi, Bengali, Marathi, Gujarati) with health insurance topics → always "yes"
3. Short unclear phrases or filler words → always "yes" (give benefit of the doubt)
4. Only say "no" if you are completely certain the topic has nothing to do with health insurance
 
Reply with ONLY one word: yes or no"""
 
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=5,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query}
            ]
        )
        answer = response.choices[0].message.content.strip().lower()
 
        # Sanitize — only accept yes/no, default to yes for safety
        if answer not in ("yes", "no"):
            print(f"[Classifier] Unexpected response '{answer}' — defaulting to yes")
            answer = "yes"
 
        print(f"[Classifier] Query: '{query}' → {answer}")
        return answer == "yes"
 
    except Exception as e:
        # If classifier fails, default to yes — better to attempt RAG than reject
        print(f"[Classifier] Error: {e} — defaulting to yes")
        return True
             
