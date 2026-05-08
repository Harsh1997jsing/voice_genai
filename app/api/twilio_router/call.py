from fastapi import APIRouter, Request
from fastapi.responses import Response
from app.services.twilio_service import make_call
from app.core.config import settings

router = APIRouter()


@router.post("/call_single")
def call_user(phone: str):
    sid = make_call(phone)
    return {"call_sid": sid}


@router.post("/twilio/voice")
async def incoming_call(request: Request):
    # FIX #4: Use your real public domain, not 127.0.0.1
    # Set PUBLIC_BASE_URL in your .env e.g. https://yourdomain.com or ngrok URL
    base_url = settings.PUBLIC_BASE_URL.rstrip("/")

    response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Hello, connecting you to AI assistant</Say>
    <Connect>
        <Stream url="wss://{base_url.replace('https://', '')}/ws/call" />
    </Connect>
</Response>"""

    return Response(content=response, media_type="application/xml")