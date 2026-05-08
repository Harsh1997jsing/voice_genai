# import asyncio
# import json
# import base64
# from fastapi import APIRouter, WebSocket
# from app.services.stt_service import open_stt, send_audio, receive_transcript
# from app.services.tts_service import open_elevenlabs_stream, send_text, pump_audio, flush_stream
# from app.services.rag_pipeline import rag_stream

# router = APIRouter()


# async def handle_transcripts(stt_ws, websocket: WebSocket, tts_ws_holder: dict):
#     async for result in receive_transcript(stt_ws):
#         if result["type"] == "final":
#             user_text = result["text"]

#             if not user_text.strip():
#                 continue

#             print(f"Final transcript: {user_text}")

#             tts_ws = tts_ws_holder.get("ws")
#             if tts_ws is None:
#                 print("[WARN] TTS not ready yet, skipping")
#                 continue

#             async for chunk in rag_stream(user_text, user_id=1):
#                 await send_text(connection=tts_ws, text=chunk)

#             await flush_stream(tts_ws)


# @router.websocket("/ws/call")
# async def voice_call(websocket: WebSocket):
#     print("WebSocket connection initiated")
#     await websocket.accept()

#     stt_ws = None
#     tts_ws = None
#     tts_ws_holder = {"ws": None}
#     stt_task = None
#     stream_sid = None

#     try:
#         while True:
#             data = await websocket.receive()

#             if "text" in data:
#                 msg = json.loads(data["text"])
#                 event = msg.get("event")

#                 if event == "start":
#                     stream_sid = msg["start"]["streamSid"]
#                     print(f"Call started: {msg}")

#                     tts_ws = await open_elevenlabs_stream(
#                         system_prompt="You are a helpful AI assistant"
#                     )
#                     tts_ws_holder["ws"] = tts_ws

#                     # FIX: Twilio requires audio as JSON text with base64 payload,
#                     # NOT raw bytes. Build a proper Twilio media message callback.
#                     async def send_audio_to_twilio(audio_bytes: bytes):
#                         payload = base64.b64encode(audio_bytes).decode("utf-8")
#                         await websocket.send_text(json.dumps({
#                             "event": "media",
#                             "streamSid": stream_sid,
#                             "media": {"payload": payload}
#                         }))

#                     asyncio.create_task(
#                         pump_audio(tts_ws, send_audio_to_twilio)
#                     )

#                 elif event == "media":
#                     audio_bytes = base64.b64decode(msg["media"]["payload"])

#                     if stt_ws is None:
#                         stt_ws = await open_stt()
#                         stt_task = asyncio.create_task(
#                             handle_transcripts(stt_ws, websocket, tts_ws_holder)
#                         )
#                         stt_task.add_done_callback(
#                             lambda t: print(
#                                 f"STT task ended: {t.exception()}"
#                                 if not t.cancelled() and t.exception()
#                                 else "STT task ended cleanly"
#                             )
#                         )

#                     await send_audio(stt_ws, audio_bytes)

#                 elif event == "stop":
#                     print("Call stopped by Twilio")
#                     break

#     finally:
#         if stt_task:
#             stt_task.cancel()
#         if stt_ws:
#             await stt_ws.close()
#         if tts_ws:
#             await tts_ws.close()


import asyncio
import json
import base64
import time
import uuid
from contextlib import suppress
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
from app.services.stt_service import open_stt, send_audio, receive_transcript
from app.services.tts_service import open_elevenlabs_stream, send_text, pump_audio, flush_stream
from app.services.rag_pipeline import rag_stream

router = APIRouter()

SAMPLE_RATE     = 8000
CHUNK_MS        = 20
BYTES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_INTERVAL  = CHUNK_MS / 1000                       
STT_BATCH_FRAMES = 1


async def tts_keepalive_loop(tts_ws_holder: dict, tts_state: dict):
    # ElevenLabs terminates stream if no text arrives for ~20s.
    # Send lightweight keepalive text periodically while call is active.
    while tts_state.get("running", True):
        await asyncio.sleep(10)
        ws = tts_ws_holder.get("ws")
        if ws is None:
            continue
        try:
            await ws.send(json.dumps({
                "text": " ",
                "try_trigger_generation": False,
            }))
        except Exception:
            # Connection may be closed; next send path will recreate.
            tts_ws_holder["ws"] = None


async def handle_transcripts(stt_ws, tts_ws_holder: dict, stt_control: dict, tts_state: dict):
    async for result in receive_transcript(stt_ws):
        if result["type"] == "final":
            user_text = result["text"]
            stt_final_ts = result.get("ts", time.perf_counter())
            if not user_text.strip():
                continue

            tts_ws = tts_ws_holder.get("ws")
            if tts_ws is None:
                print("[TTS] Stream unavailable, skipping response")
                continue

            try:
                tts_state["speaking"] = True
                tts_state["started_at"] = asyncio.get_running_loop().time()
                # Pause STT ingestion while assistant is speaking (half-duplex).
                stt_control["paused"] = True
                trace_id = f"{tts_state.get('call_trace', 'call')}-{tts_state.get('turn', 0)}"
                print(f"[LATENCY] trace={trace_id} stage=stt_final")
                buffer = ""
                sent_first_tts_chunk = False
                async for chunk in rag_stream(user_text, user_id=1, trace_id=trace_id):
                    buffer += chunk
                    if not sent_first_tts_chunk:
                        first_tts_ms = (time.perf_counter() - stt_final_ts) * 1000
                        print(f"[LATENCY] trace={trace_id} stage=stt_to_first_tts_text ms={first_tts_ms:.1f}")
                        sent_first_tts_chunk = True
                    await send_text(connection=tts_ws, text=chunk)

               
                    if any(buffer.rstrip().endswith(p) for p in [".", "!", "?", ":", "\n"]):
                        await flush_stream(tts_ws)
                        buffer = ""

                # Final flush for whatever is left
                if buffer.strip():
                    await flush_stream(tts_ws)
                total_turn_ms = (time.perf_counter() - stt_final_ts) * 1000
                print(f"[LATENCY] trace={trace_id} stage=tts_flush_done ms={total_turn_ms:.1f}")

            except Exception as e:
                print(f"[RAG/TTS] Error: {e}")
            finally:
                tts_state["speaking"] = False
                tts_state["started_at"] = 0.0
                stt_control["paused"] = False
                tts_state["turn"] = tts_state.get("turn", 0) + 1


@router.websocket("/ws/call")
async def voice_call(websocket: WebSocket):
    print("WebSocket connection initiated")
    await websocket.accept()

    stt_ws        = None
    tts_ws        = None
    tts_ws_holder = {"ws": None}
    stt_control   = {"paused": False}
    tts_state     = {
        "speaking": False,
        "running": True,
        "started_at": 0.0,
        "call_trace": f"call-{uuid.uuid4().hex[:8]}",
        "turn": 1,
    }
    stt_task      = None
    stt_broken    = False
    tts_keepalive_task = None
    tts_pump_task = None
    stream_sid    = None
    audio_buffer  = bytearray()
    frame_count   = 0

    async def send_audio_to_twilio(audio_bytes: bytes):
        for i in range(0, len(audio_bytes), BYTES_PER_CHUNK):
            chunk   = audio_bytes[i:i + BYTES_PER_CHUNK]
            payload = base64.b64encode(chunk).decode("utf-8")
            try:
                await websocket.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": stream_sid,
                    "media":     {"payload": payload}
                }))
            except Exception as e:
                print(f"[TTS] Twilio send failed, stopping pump: {e}")
                return
            await asyncio.sleep(CHUNK_INTERVAL)

    async def start_tts_stream():
        nonlocal tts_pump_task
        ws = await open_elevenlabs_stream(
            system_prompt="I am NovaCare Health Insurance AI assistant."
        )
        tts_ws_holder["ws"] = ws
        tts_pump_task = asyncio.create_task(pump_audio(ws, send_audio_to_twilio))

    try:
        while True:
            try:
                data = await websocket.receive()
            except (WebSocketDisconnect, ConnectionClosed):
                print("Twilio WebSocket disconnected")
                break

            if "text" not in data:
                continue

            msg   = json.loads(data["text"])
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                print(f"Call started: {msg}")

                await start_tts_stream()
                if tts_keepalive_task is None:
                    tts_keepalive_task = asyncio.create_task(
                        tts_keepalive_loop(tts_ws_holder, tts_state)
                    )

            elif event == "media":
                raw_bytes = base64.b64decode(msg["media"]["payload"])

                if stt_ws is None:
                    stt_ws   = await open_stt()
                    stt_task = asyncio.create_task(
                        handle_transcripts(stt_ws, tts_ws_holder, stt_control, tts_state)
                    )
                    stt_task.add_done_callback(
                        lambda t: print(
                            f"STT task ended: {t.exception()}"
                            if not t.cancelled() and t.exception()
                            else "STT task ended cleanly"
                        )
                    )

                if stt_broken:
                    # STT connection is unusable (e.g. insufficient funds). Skip sends.
                    continue

                if stt_control["paused"]:
                    # Half-duplex mode: ignore caller audio while TTS is speaking.
                    continue

                audio_buffer.extend(raw_bytes)
                frame_count += 1

                if frame_count >= STT_BATCH_FRAMES:
                    try:
                        await send_audio(stt_ws, bytes(audio_buffer))
                    except Exception as e:
                        err = str(e)
                        if "insufficient_funds_initial_check" in err:
                            print("[STT] Disabled for this call: insufficient ElevenLabs balance")
                        else:
                            print(f"[STT] send_audio failed, disabling STT for this call: {e}")
                        stt_broken = True
                    finally:
                        audio_buffer.clear()
                        frame_count = 0

            elif event == "stop":
                print("Call stopped by Twilio")
                break

    finally:
        tts_state["running"] = False
        if stt_task:
            stt_task.cancel()
        if tts_keepalive_task:
            tts_keepalive_task.cancel()
            with suppress(asyncio.CancelledError):
                await tts_keepalive_task
        if tts_pump_task:
            tts_pump_task.cancel()
            with suppress(asyncio.CancelledError):
                await tts_pump_task
        if stt_ws:
            try:
                await stt_ws.close()
            except Exception:
                pass
        tts_ws = tts_ws_holder.get("ws")
        if tts_ws:
            try:
                await tts_ws.close()
            except Exception:
                pass
        print("Call cleanup done")
