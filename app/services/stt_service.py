# import json
# import base64
# from websockets.asyncio.client import connect as ws_connect
# from app.core.config import settings

# ELEVEN_API_KEY = settings.ELEVENLABS_API_KEY

# # ulaw_8000 is valid per docs and matches Twilio natively — no conversion needed
# STT_URL = (
#     "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
#     "?model_id=scribe_v2_realtime"
#     "&audio_format=ulaw_8000"
#     "&commit_strategy=vad"
#     "&vad_silence_threshold_secs=0.3"
# )


# async def open_stt():
#     ws = await ws_connect(
#         STT_URL,
#         additional_headers={"xi-api-key": ELEVEN_API_KEY},
#     )

#     # ElevenLabs always sends session_started as the very first message
#     # We MUST read and handle it before doing anything else
#     raw = await ws.recv()
#     data = json.loads(raw)

#     if data["message_type"] == "session_started":
#         print(f"[STT] Session started. Config: {data.get('config')}")

#     elif data["message_type"] == "auth_error":
#         raise Exception(f"[STT] Auth failed: {data.get('error')} — check your ELEVENLABS_API_KEY")

#     elif data["message_type"] == "error":
#         raise Exception(f"[STT] Connection error: {data.get('error')}")

#     else:
#         # Log unexpected first message so we can debug
#         raise Exception(f"[STT] Unexpected first message: {data}")

#     return ws


# async def send_audio(ws, audio_bytes: bytes):
#     # Send Twilio mulaw 8kHz directly — no conversion needed with ulaw_8000 format
#     await ws.send(json.dumps({
#         "message_type": "input_audio_chunk",
#         "audio_base_64": base64.b64encode(audio_bytes).decode(),
#         "commit": False,
#         "sample_rate": 8000,
#     }))


# async def receive_transcript(ws):
#     async for msg in ws:
#         data = json.loads(msg)
#         mtype = data.get("message_type")

#         if mtype == "partial_transcript":
#             yield {"type": "partial", "text": data.get("text", "")}

#         elif mtype == "committed_transcript":
#             yield {"type": "final", "text": data.get("text", "")}

#         elif mtype in ("error", "auth_error", "quota_exceeded",
#                        "rate_limited", "input_error", "transcriber_error"):
#             # Log real ElevenLabs error so you can see what's wrong
#             print(f"[STT] Error from ElevenLabs ({mtype}): {data.get('error')}")
#             break

#         # All other types (session_started already handled in open_stt) — skip silently


import json
import base64
import time
from websockets.asyncio.client import connect as ws_connect
from app.core.config import settings

ELEVEN_API_KEY = settings.ELEVENLABS_API_KEY

# NOTE: ElevenLabs STT does NOT support vad_silence_threshold_secs as a URL param.
# VAD config must be passed as a JSON message after session_started.
STT_URL = (
    "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    "?model_id=scribe_v2_realtime"
    "&audio_format=ulaw_8000"
    "&commit_strategy=vad"
)


async def open_stt():
    ws = await ws_connect(
        STT_URL,
        additional_headers={"xi-api-key": ELEVEN_API_KEY},
    )

    raw  = await ws.recv()
    data = json.loads(raw)

    if data["message_type"] == "session_started":
        print(f"[STT] Session started. Config: {data.get('config')}")
    elif data["message_type"] == "auth_error":
        raise Exception(f"[STT] Auth failed: {data.get('error')}")
    elif data["message_type"] == "error":
        raise Exception(f"[STT] Connection error: {data.get('error')}")
    else:
        raise Exception(f"[STT] Unexpected first message: {data}")

    return ws


async def send_audio(ws, audio_bytes: bytes):
    await ws.send(json.dumps({
        "message_type": "input_audio_chunk",
        "audio_base_64": base64.b64encode(audio_bytes).decode(),
        "commit":        False,
        "sample_rate":   8000,
    }))


async def receive_transcript(ws):
    async for msg in ws:
        data  = json.loads(msg)
        mtype = data.get("message_type")

        if mtype == "partial_transcript":
            yield {
                "type": "partial",
                "text": data.get("text", ""),
                "ts": time.perf_counter(),
            }

        elif mtype == "committed_transcript":
            yield {
                "type": "final",
                "text": data.get("text", ""),
                "ts": time.perf_counter(),
            }

        elif mtype in ("error", "auth_error", "quota_exceeded",
                       "rate_limited", "input_error", "transcriber_error",
                       "queue_overflow"):
            print(f"[STT] ElevenLabs error ({mtype}): {data.get('error')}")
            break
