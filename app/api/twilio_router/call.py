import asyncio
from urllib.parse import urlencode
from xml.sax.saxutils import escape

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import Response

from app.core.config import settings
from app.core.deps import get_current_user
from app.db.database import get_db
from app.models import leads
from app.models.phone_number import PhoneNumber
from app.models.promp_model import Prompt
from app.services.twilio_service import make_call

router = APIRouter()
BATCH_SIZE = 5


# ─────────────────────────────────────────────────────────────────────────────
# Single call
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/call_single")
async def call_single(                              # FIX 1: must be async
    phone: str,
    current_user=Depends(get_current_user),
    db=Depends(get_db),
):
    # Look up the latest prompt for this user
    latest_prompt = (
        db.query(Prompt)
        .filter(Prompt.call_id.like(f"call_{current_user.id}_%"))
        .order_by(Prompt.created_at.desc())
        .first()
    )
    if not latest_prompt:
        raise HTTPException(
            status_code=404,
            detail="No prompt found for this user. Upload company PDF first.",
        )

    twilio_number = (
        db.query(PhoneNumber)
        .filter(PhoneNumber.user_id == current_user.id)
        .first()
    )
    if not twilio_number:
        raise HTTPException(
            status_code=400,
            detail="User does not have a Twilio number. Please buy one first.",
        )

    one_liner    = (latest_prompt.one_liner     or {}).get("text", "")
    system_prompt = (latest_prompt.system_prompt or {}).get("text", "")

    # FIX 2: run the blocking Twilio HTTP call in a thread pool
    sid = await asyncio.to_thread(
        make_call,
        to_number=phone,
        user_id=current_user.id,
        one_liner=one_liner,
        system_prompt=system_prompt,
        from_number=twilio_number.number,
    )
    return {"call_sid": sid}


# ─────────────────────────────────────────────────────────────────────────────
# Bulk calls
# ─────────────────────────────────────────────────────────────────────────────

async def _start_call(
    phone: str,
    user_id: int,
    one_liner: str,
    system_prompt: str,
    from_number: str | None,
) -> str:
    """Thin async wrapper so asyncio.gather can await each Twilio call."""
    # FIX 3: make_call is sync → wrap in to_thread
    return await asyncio.to_thread(
        make_call,
        to_number=phone,
        user_id=user_id,
        one_liner=one_liner,
        system_prompt=system_prompt,
        from_number=from_number,
    )


@router.post("/call_bulk")
async def call_bulk(
    current_user=Depends(get_current_user),         # FIX 4: correct dep type
    db=Depends(get_db),
):
    # FIX 5: query DB once, not once-per-phone inside a comprehension
    twilio_number = (
        db.query(PhoneNumber)
        .filter(PhoneNumber.user_id == current_user.id)
        .first()
    )
    from_num = twilio_number.number if twilio_number else None

    latest_prompt = (
        db.query(Prompt)
        .filter(Prompt.call_id.like(f"call_{current_user.id}_%"))
        .order_by(Prompt.created_at.desc())
        .first()
    )
    one_liner     = (latest_prompt.one_liner     or {}).get("text", "") if latest_prompt else ""
    system_prompt = (latest_prompt.system_prompt or {}).get("text", "") if latest_prompt else ""

    # FIX 6: use current_user.id (int) for the leads filter
    phones = (
        db.query(leads)
        .filter(leads.c.user_id == current_user.id)
        .all()
    )

    results = []
    for i in range(0, len(phones), BATCH_SIZE):
        batch = phones[i : i + BATCH_SIZE]

        tasks = [
            _start_call(
                phone=p.phone_number,
                user_id=current_user.id,
                one_liner=one_liner,
                system_prompt=system_prompt,
                from_number=from_num,
            )
            for p in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(batch_results)
        print(f"Batch completed: {i} → {i + len(batch)}")

    return {
        "calls_started": len(results),
        "call_sids": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Twilio webhook  →  returns TwiML that opens the WebSocket stream
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/twilio/voice")
async def twilio_voice_webhook(request: Request, db=Depends(get_db)):
    """
    Twilio calls this URL after the outbound call connects.
    We reply with TwiML that:
      1. Optionally speaks the one-liner greeting
      2. Opens a <Stream> WebSocket back to /ws/call
         with user_id, system_prompt, one_liner forwarded as query params.

    WebSocket param check
    ─────────────────────
    The /ws/call handler reads these query params:
        ?user_id=<int>&system_prompt=<url-encoded>&one_liner=<url-encoded>

    user_id      — passed as plain int string, parsed with int() in ws handler
    system_prompt — URL-encoded with quote_plus; ws handler reads raw string (no decode needed,
                       WebSocket framework decodes automatically)
    one_liner    — same as system_prompt

    All three params match exactly what voice_call() in ws_routes.py expects.
    """
    base_url      = settings.PUBLIC_BASE_URL.rstrip("/")
    user_id = request.query_params.get("user_id", "").strip()

    # Build the wss:// URL for <Stream>
    # Strip the scheme so we get just the host+path
    host       = base_url.replace("https://", "").replace("http://", "")
    stream_url = f"wss://{host}/ws/call"

    if user_id:
        stream_url = f"{stream_url}?{urlencode({'user_id': user_id})}"

    # say_block = f"<Say>{escape(one_liner)}</Say>" if one_liner else ""

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")
