import asyncio
import base64
import json
import time
import uuid
from contextlib import suppress

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from app.db.database import SessionLocal
from app.models.promp_model import Prompt
from app.services.calling.constants import (
    BYTES_PER_CHUNK,
    CHUNK_INTERVAL,
    MAX_CALL_DURATION_SEC,
    SILENCE_TIMEOUT,
    STT_BATCH_FRAMES,
    SYSTEM_PROMPT,
    VAD_CHECK_EVERY,
    call_state,
)
from app.services.calling.transcript_flow import handle_transcripts, tts_keepalive_loop
from app.services.calling.vad_audio import detect_speech_async, mulaw_to_float32
from app.services.stt_service import open_stt, send_audio
from app.services.transcript_db import save_call_transcript_to_db_sync
from app.services.transcript_service import transcript_db_worker
from app.services.tts_service import open_elevenlabs_stream, pump_audio

log = structlog.get_logger()


def _load_user_prompts(current_user_id: int) -> tuple[str, str]:
    db = SessionLocal()
    try:
        latest_prompt = (
            db.query(Prompt)
            .filter(Prompt.call_id.like(f"call_{current_user_id}_%"))
            .order_by(Prompt.created_at.desc())
            .first()
        )
        if not latest_prompt:
            return "", ""
        system_prompt = (latest_prompt.system_prompt or {}).get("text", "").strip()
        one_liner = (latest_prompt.one_liner or {}).get("text", "").strip()
        return system_prompt, one_liner
    finally:
        db.close()


async def handle_voice_call(websocket: WebSocket):
    await websocket.accept()

    call_id = uuid.uuid4().hex[:8]
    call_log = log.bind(call_id=call_id)

    user_id_raw = websocket.query_params.get("user_id", "1")
    try:
        user_id = int(user_id_raw)
    except ValueError:
        user_id = 1

    dynamic_system_prompt, dynamic_one_liner = _load_user_prompts(user_id)
    call_log.info("call_accepted")

    transcript_queue = asyncio.Queue()
    db_worker_task = asyncio.create_task(transcript_db_worker(transcript_queue))

    stt_ws = None
    tts_ws_holder: dict = {"ws": None}
    tts_lock = asyncio.Lock()
    stt_control: dict = {"paused": False}
    tts_state: dict = {
        "speaking": False,
        "running": True,
        "call_trace": f"call-{call_id}",
        "turn": 1,
    }
    vad_state: dict = {"speaking": False}

    stt_task = None
    tts_keepalive_task = None
    tts_pump_task = None
    stream_sid = None
    audio_buffer = bytearray()
    frame_count = 0
    vad_counter = 0
    last_voice_time = time.perf_counter()
    call_start_time = time.perf_counter()
    stt_max_retries = 3
    stt_retry_delays = [0.5, 1.0, 2.0]
    stt_open_attempts = 0
    stt_open_failed = False

    conversation_history = []

    def get_tts_pump_task():
        return tts_pump_task

    def set_tts_pump_task(task):
        nonlocal tts_pump_task
        tts_pump_task = task

    async def send_audio_to_twilio(audio_bytes: bytes):
        start = asyncio.get_event_loop().time()
        chunks = [audio_bytes[i:i + BYTES_PER_CHUNK]
                  for i in range(0, len(audio_bytes), BYTES_PER_CHUNK)]

        for idx, chunk in enumerate(chunks):
            payload = base64.b64encode(chunk).decode("utf-8")
            try:
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }))
            except Exception as e:
                call_log.warning("twilio_send_failed", error=str(e))
                return

            # Time-anchored pacing — prevents drift accumulation
            target_time = start + (idx + 1) * CHUNK_INTERVAL
            now = asyncio.get_event_loop().time()
            if target_time > now:
                await asyncio.sleep(target_time - now)

    async def start_tts_stream():
        nonlocal tts_pump_task
        ws = await open_elevenlabs_stream(system_prompt=dynamic_one_liner or SYSTEM_PROMPT)
        async with tts_lock:
            tts_ws_holder["ws"] = ws

        async def pump_and_reset():
            try:
                await pump_audio(ws, send_audio_to_twilio)
            finally:
                async with tts_lock:
                    if tts_ws_holder.get("ws") is ws:
                        tts_ws_holder["ws"] = None
                call_log.info("tts_pump_exited")

        tts_pump_task = asyncio.create_task(pump_and_reset())
        set_tts_pump_task(tts_pump_task)

    async def restart_tts_stream():
        nonlocal tts_pump_task
        call_log.info("barge_in_tts_restart")
        old_pump = tts_pump_task
        if old_pump and not old_pump.done():
            old_pump.cancel()
            with suppress(asyncio.CancelledError):
                await old_pump

        async with tts_lock:
            old_ws = tts_ws_holder.get("ws")
            tts_ws_holder["ws"] = None

        if old_ws:
            with suppress(Exception):
                await old_ws.close()

        await start_tts_stream()

    async def handle_barge_in():
        if tts_state["speaking"]:
            tts_state["speaking"] = False
            stt_control["paused"] = False
            await restart_tts_stream()

    try:
        while True:
            if time.perf_counter() - call_start_time > MAX_CALL_DURATION_SEC:
                call_log.warning("max_call_duration_reached")
                break

            if call_state.get("ending"):
                call_log.info("call_ending_by_phrase")
                break

            try:
                data = await websocket.receive()
                if data["type"] == "websocket.disconnect":
                    call_log.info("twilio_disconnect_event")
                    break
            except (WebSocketDisconnect, ConnectionClosed, RuntimeError):
                call_log.info("twilio_disconnected")
                break

            if "text" not in data:
                continue

            msg = json.loads(data["text"])
            event = msg.get("event")

            if event == "start":
                stream_sid = msg["start"]["streamSid"]

                call_sid   = msg["start"].get("callSid") 
                call_log.info("call_started", stream_sid=stream_sid)
                await start_tts_stream()
                if tts_keepalive_task is None:
                    tts_keepalive_task = asyncio.create_task(
                        tts_keepalive_loop(tts_ws_holder, tts_lock, tts_state)
                    )

            elif event == "media":
                raw_bytes = base64.b64decode(msg["media"]["payload"])
                vad_counter += 1
                if vad_counter >= VAD_CHECK_EVERY:
                    vad_counter = 0
                    audio_float = mulaw_to_float32(raw_bytes)
                    is_speaking = await detect_speech_async(audio_float)
                else:
                    is_speaking = vad_state["speaking"]

                now = time.perf_counter()
                if is_speaking:
                    if tts_state["speaking"]:
                        barge_in_task = asyncio.create_task(handle_barge_in())

                        def _barge_in_done(fut: asyncio.Task):
                            if not fut.cancelled() and fut.exception():
                                call_log.error("barge_in_error", error=str(fut.exception()))

                        barge_in_task.add_done_callback(_barge_in_done)

                    last_voice_time = now
                    if not vad_state["speaking"]:
                        call_log.debug("speech_started")
                    vad_state["speaking"] = True
                else:
                    silence_duration = now - last_voice_time
                    if vad_state["speaking"] and silence_duration > SILENCE_TIMEOUT:
                        call_log.debug("speech_ended", silence_sec=round(silence_duration, 2))
                        vad_state["speaking"] = False

                if stt_ws is None and not stt_open_failed:
                    if stt_open_attempts >= stt_max_retries:
                        stt_open_failed = True
                        call_log.error("stt_max_retries_exceeded")
                        continue

                    try:
                        stt_open_attempts += 1
                        if stt_open_attempts > 1:
                            delay = stt_retry_delays[min(stt_open_attempts - 2, len(stt_retry_delays) - 1)]
                            await asyncio.sleep(delay)

                        call_log.info("stt_opening", attempt=stt_open_attempts)
                        stt_ws = await open_stt()
                        stt_open_attempts = 0
                        stt_task = asyncio.create_task(
                            handle_transcripts(
                                stt_ws,
                                tts_ws_holder,
                                tts_lock,
                                stt_control,
                                tts_state,
                                vad_state,
                                call_log,
                                user_id,
                                send_audio_to_twilio,
                                get_tts_pump_task,
                                set_tts_pump_task,
                                conversation_history,
                                dynamic_system_prompt,
                                call_sid=call_sid,
                            )
                        )
                    except Exception as e:
                        call_log.error("stt_open_failed", error=str(e))

                # Always forward audio to STT even while paused
                # to prevent the 20-second inactivity timeout (1008).
                # Transcripts received during pause are harmless — 
                # handle_transcripts won't act while tts_state["speaking"] is True.
                audio_buffer.extend(raw_bytes)
                frame_count += 1
                if frame_count >= STT_BATCH_FRAMES:
                    try:
                        await send_audio(stt_ws, bytes(audio_buffer))
                    except Exception as e:
                        call_log.warning("stt_send_failed", error=str(e))
                        # Dead STT connection — null it out so the
                        # reconnect logic on line 231 triggers next frame.
                        stt_ws = None
                        if stt_task and not stt_task.done():
                            stt_task.cancel()
                    finally:
                        audio_buffer.clear()
                        frame_count = 0

            elif event == "stop":
                call_log.info("call_stopped")
                break

    finally:
        await transcript_queue.join()
        db_worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await db_worker_task

        tts_state["running"] = False
        call_log.info("call_cleanup_start")

        for task in (stt_task, tts_keepalive_task, tts_pump_task):
            if task:
                task.cancel()

        for task in (tts_keepalive_task, tts_pump_task):
            if task:
                with suppress(asyncio.CancelledError):
                    await task

        if stt_ws:
            with suppress(Exception):
                await stt_ws.close()

        async with tts_lock:
            tts_ws = tts_ws_holder.get("ws")
        if tts_ws:
            with suppress(Exception):
                await tts_ws.close()

        await asyncio.to_thread(
            save_call_transcript_to_db_sync,
            tts_state["call_trace"],
            conversation_history,
        )

        call_log.info("call_cleanup_done")
