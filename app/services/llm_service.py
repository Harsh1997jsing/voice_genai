from openai import AsyncOpenAI
from app.core.config import settings
import time
import asyncio

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

 
LLM_MODEL       = "gpt-4o-mini"
LLM_MAX_TOKENS  = 80      # 1-2 sentences max — this is a voice call
LLM_TEMPERATURE = 0.2     # low = faster + deterministic
 
 
SYSTEM_PROMPT_BASE = """\
You are Nova, a NovaCare Health Insurance voice assistant on a live phone call.
 
STRICT RULES:
1. Answer ONLY using the context provided. Do NOT use any outside knowledge.
2. If the answer is not in the context, say exactly: "I don't have that information right now."
3. Reply in the SAME language the user used (Hindi, Punjabi, Malayalam, Bengali, English, etc.)
4. Keep answers SHORT — 1 to 2 sentences only. This is a voice call, not a chat.
5. NEVER use bullet points, numbers, markdown, or lists in your response.
6. NEVER repeat the question back to the user.
7. Speak naturally, as a human agent would on a phone call."""
 
 
def build_messages(query: str, context: str) -> list:
    """
    Build the messages array for the LLM.
    System prompt carries context + rules.
    User message is the clean query only.
    """
    system_content = (
        f"{SYSTEM_PROMPT_BASE}\n\n"
        f"--- CONTEXT START ---\n"
        f"{context}\n"
        f"--- CONTEXT END ---"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": query},
    ]
 

 
async def stream_llm(
    query: str,
    context: str,
    trace_id: str = "na"
):
    """
    Stream LLM response token by token.
 
    Optimizations applied:
    - max_tokens=80        → short answers, less generation time
    - temperature=0.2      → faster + consistent
    - system/user split    → cleaner context injection
    - no retries           → fail fast for voice latency
    """
 
    t0 = time.perf_counter()
 
    first_token_logged = False
 
    try:
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=build_messages(query, context),
            # max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stream=True,
        )
 
        async for chunk in stream:
 
            delta = chunk.choices[0].delta.content
 
            if not delta:
                continue
 
            # Log first token latency once
            if not first_token_logged:
                first_token_ms = (time.perf_counter() - t0) * 1000
                print(
                    f"[LATENCY] trace={trace_id} "
                    f"stage=llm_first_token ms={first_token_ms:.1f}"
                )
                first_token_logged = True
 
            yield delta
 
    except asyncio.CancelledError:
        # Caller cancelled (e.g. barge-in) — exit cleanly
        print(f"[LLM] Stream cancelled: trace={trace_id}")
        return
 
    except Exception as e:
        print(f"[LLM] Error: trace={trace_id} error={e}")
 
        # Yield a safe fallback so TTS always has something to say
        yield "I'm having trouble right now. Please try again."
        return
 
    finally:
        total_ms = (time.perf_counter() - t0) * 1000
        print(
            f"[LATENCY] trace={trace_id} "
            f"stage=llm_total ms={total_ms:.1f}"
        )
 



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
            # max_tokens=,
            temperature=0.2,
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
             
