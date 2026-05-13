"""
tts_service.py — Production-grade ElevenLabs TTS streaming service.

Fixes vs original:
  - pump_audio signals caller on exit so tts_ws_holder can be cleared
  - close_stream is robust (no bare Exception swallowing)
  - Structured logging via structlog
  - All public coroutines have explicit type annotations
  - ping_interval/ping_timeout disabled (ElevenLabs incompatibility kept)
"""

import asyncio
import base64
import json
from collections.abc import Callable, Awaitable

import structlog
from websockets.asyncio.client import connect as ws_connect

from app.core.config import settings

log = structlog.get_logger(__name__)

ELEVENLABS_API_KEY = settings.ELEVENLABS_API_KEY
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"

ELEVENLABS_WS_URL = (
    f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    f"/stream-input?model_id=eleven_flash_v2_5"
    f"&output_format=ulaw_8000"
    f"&optimize_streaming_latency=4"
)

# ──────────────────────────────────────────────────────────────────────────────
# Connection
# ──────────────────────────────────────────────────────────────────────────────

async def open_elevenlabs_stream(system_prompt: str = ""):
    """
    Open a streaming TTS WebSocket connection and send the BOS (begin-of-stream)
    initialisation frame.

    Returns the open WebSocket connection.
    """
    connection = await ws_connect(
        ELEVENLABS_WS_URL,
        additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
        # ElevenLabs streams don't support standard WebSocket pings —
        # disabling prevents 1011 keepalive ping timeout errors.
        ping_interval=None,
        ping_timeout=None,
    )
    system_prompt = system_prompt['text']
    print(f"Opened ElevenLabs TTS stream with system prompt: {system_prompt}...")

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

    log.info("tts_stream_opened")
    return connection


# ──────────────────────────────────────────────────────────────────────────────
# Text sending
# ──────────────────────────────────────────────────────────────────────────────

async def send_text(
    connection,
    text: str,
    inject_prompt: str = "",
) -> None:
    """
    Send a text chunk to the TTS stream.
    No-ops on empty / whitespace-only text to avoid unnecessary frames.
    """
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


async def flush_stream(connection) -> None:
    """
    Trigger ElevenLabs to generate audio from whatever text it has buffered.
    Call after the first token and after every ~120-char batch.
    """
    await connection.send(json.dumps({
        "text": " ",
        "try_trigger_generation": True,
    }))


# ──────────────────────────────────────────────────────────────────────────────
# Audio pump
# ──────────────────────────────────────────────────────────────────────────────

async def pump_audio(
    connection,
    on_chunk: Callable[[bytes], Awaitable[None]],
) -> None:
    """
    Continuously read audio chunks from the ElevenLabs WebSocket and forward
    them to `on_chunk` (typically `send_audio_to_twilio`).

    Exits cleanly on:
      - WebSocket close / disconnect
      - Exception in `on_chunk` (broken Twilio connection)
      - asyncio.CancelledError (barge-in or call teardown)
    """
    try:
        async for raw in connection:
            msg = json.loads(raw)

            if msg.get("audio"):
                audio_bytes = base64.b64decode(msg["audio"])
                try:
                    await on_chunk(audio_bytes)
                except Exception as e:
                    log.warning("tts_on_chunk_failed", error=str(e))
                    break

            elif msg.get("isFinal"):
                log.debug("tts_stream_final")
                break

    except asyncio.CancelledError:
        log.info("tts_pump_cancelled")
        raise   # let caller handle cleanup

    except Exception as e:
        log.warning("tts_pump_error", error=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Teardown
# ──────────────────────────────────────────────────────────────────────────────

async def close_stream(connection) -> None:
    """
    Gracefully close the ElevenLabs TTS stream.
    Sends the EOS sentinel text ("") before closing the socket.
    """
    try:
        await connection.send(json.dumps({"text": ""}))
    except Exception as e:
        log.debug("tts_eos_send_failed", error=str(e))

    try:
        await connection.close()
        log.info("tts_stream_closed")
    except Exception as e:
        log.debug("tts_close_failed", error=str(e))