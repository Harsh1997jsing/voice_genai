import asyncio
import json
import base64
import os
from websockets.asyncio.client import connect as ws_connect
from app.core.config import settings

ELEVENLABS_API_KEY = settings.ELEVENLABS_API_KEY
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

ELEVENLABS_WS_URL = (
    f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    f"/stream-input?model_id=eleven_flash_v2_5"
    f"&output_format=ulaw_8000"
    f"&optimize_streaming_latency=4"
)


async def open_elevenlabs_stream(system_prompt: str = ""):
    connection = await ws_connect(
        ELEVENLABS_WS_URL,
        additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
        # FIX: Disable websockets auto ping — it conflicts with ElevenLabs
        # and causes 1011 keepalive ping timeout errors
        ping_interval=None,
        ping_timeout=None,
    )

    bos_text = f"{system_prompt.strip()} " if system_prompt.strip() else " "

    await connection.send(json.dumps({
        "text": bos_text,
        "flush": True,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "use_speaker_boost": False,
        },
        "generation_config": {
            "chunk_length_schedule": [50],
        },
    }))
    return connection


async def send_text(connection, text: str, inject_prompt: str = ""):
    if not text.strip():
        return

    payload = (
        f"{text.strip()} {inject_prompt.strip()}"
        if inject_prompt.strip()
        else text
    )

    await connection.send(json.dumps({
        "text": payload,
        "try_trigger_generation": True,
    }))


async def flush_stream(connection):
    await connection.send(json.dumps({
        "text": " ",
        "try_trigger_generation": True,
    }))


async def pump_audio(connection, on_chunk):
    try:
        async for raw in connection:
            msg = json.loads(raw)
            if msg.get("audio"):
                audio_bytes = base64.b64decode(msg["audio"])
                try:
                    await on_chunk(audio_bytes)
                except Exception as e:
                    print(f"[TTS] Twilio send failed, stopping pump: {e}")
                    break
    except Exception as e:
        print(f"[TTS] pump_audio ended: {e}")


async def close_stream(connection):
    try:
        await connection.send(json.dumps({"text": ""}))
        await connection.close()
    except Exception:
        pass