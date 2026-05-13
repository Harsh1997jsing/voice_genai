import json
from app.core.llm_client import OpenAIClient

async def _generate_system_prompt(
    raw_text: str,
    company_name: str | None,
) -> dict:

    snippet = raw_text[:12000]
    company = (company_name or "the company").strip()

    system_instruction = """
You are an expert AI voice-agent prompt engineer.

Your task is to analyze company documents and generate:

1. A short natural phone-call introduction sentence.
2. A strict RAG system prompt for a realtime AI voice agent.

Return ONLY valid JSON with this exact structure:

{
  "call_intro": "...",
  "system_prompt": "..."
}
"""

    user_instruction = f"""
Company Name:
{company}

Create TWO outputs:

-----------------------------------
1. call_intro
-----------------------------------

Requirements:
- Max 1-2 short sentences
- Friendly and natural
- Used as opening line in phone call
- Mention company name naturally
- Mention core purpose/service briefly
- Avoid sounding robotic or sales-heavy

Example:
"Hi, this is Sarah from Acme Solar. We help homeowners reduce electricity costs through affordable solar solutions."

-----------------------------------
2. system_prompt
-----------------------------------

This prompt will be used by a realtime voice AI agent with RAG retrieval.

The prompt must clearly define:

IDENTITY:
- Who the assistant is
- Company representation

BEHAVIOR:
- Keep responses short and conversational
- Speak naturally for phone calls
- Ask clarifying questions
- Never hallucinate
- Never invent pricing/features/policies
- Only answer from retrieved knowledge or provided context
- If unsure, politely say information is unavailable
- Handle interruptions naturally
- Support multilingual conversations
- Stay concise

RAG RULES:
- Retrieved context is highest priority
- Do not fabricate missing details
- If retrieval is empty, admit uncertainty
- Never assume unavailable information

SAFETY:
- Do not provide legal/medical/financial guarantees unless explicitly present
- Do not make commitments on behalf of company
- Escalate sensitive requests to human support

VOICE STYLE:
- Short sentences
- Human-like
- Avoid long paragraphs
- Avoid bullet points in responses

ESCALATION:
- If user asks unsupported questions:
  "I don't have that information right now, but I can connect you with the support team."

Use ONLY information from source text.

-----------------------------------
SOURCE DOCUMENT
-----------------------------------

{snippet}
"""


    content = await OpenAIClient(model="gpt-4o-mini").complete(
        system=system_instruction,
        prompt=user_instruction,
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    if not content:
        raise RuntimeError("Prompt generation returned empty content")

    return json.loads(content)
