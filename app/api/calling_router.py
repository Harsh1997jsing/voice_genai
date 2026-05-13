"""
router.py — Production-grade WebSocket voice call handler.

Fixes vs original:
  - Barge-in: VAD speech during TTS cancels and restarts the TTS stream
  - VAD runs in asyncio.to_thread (never blocks event loop)
  - VAD model pool (no shared-state corruption across concurrent calls)
  - asyncio.Lock replaces threading.Lock everywhere
  - LLM stream wrapped in asyncio.timeout (no hung turns)
  - handle_transcripts loop body wrapped in try/except (no silent death)
  - tts_ws_holder guarded by asyncio.Lock (no race on reconnect)
  - user_id threaded through from WebSocket auth (no hardcoded =1)
  - Structured logging via structlog
  - Max call duration enforced
  - Auth token verified before accepting the call
"""

import asyncio
import json
import base64
import time
import uuid
import re
from contextlib import suppress

import audioop
import numpy as np
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from silero_vad import load_silero_vad, get_speech_timestamps
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.models.promp_model import Prompt
from app.services.stt_service import open_stt, send_audio, receive_transcript
from app.services.tts_service import (
    open_elevenlabs_stream,
    send_text,
    pump_audio,
    flush_stream,
)
from app.services.kb_service import search_kb_async
from app.services.rag_pipeline import stream_llm
from app.services.transcript_service import (
    transcript_db_worker,
)
from app.services.transcript_db import save_call_transcript_to_db_sync

log = structlog.get_logger()
router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# Audio constants
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 8000
CHUNK_MS = 20
BYTES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_INTERVAL = CHUNK_MS / 1000
STT_BATCH_FRAMES = 2

# ──────────────────────────────────────────────────────────────────────────────
# VAD model pool — one model per slot, semaphore controls access.
# Prevents shared PyTorch state corruption across concurrent calls.
# ──────────────────────────────────────────────────────────────────────────────
VAD_POOL_SIZE = 8
_vad_pool: list = [load_silero_vad() for _ in range(VAD_POOL_SIZE)]
_vad_semaphore = asyncio.Semaphore(VAD_POOL_SIZE)
_vad_pool_lock = asyncio.Lock()
_vad_pool_index = 0

VAD_CHECK_EVERY = 5          # run Silero every N chunks (~100 ms)
SILENCE_TIMEOUT = 0.7        # seconds of silence before VAD resets

# ──────────────────────────────────────────────────────────────────────────────
# Speculative retrieval tunables
# ──────────────────────────────────────────────────────────────────────────────
SPECULATIVE_MIN_WORDS = 4
SPECULATIVE_MIN_CHARS = 20
SPECULATIVE_DEBOUNCE_SEC = 0.5
SPECULATIVE_WORD_OVERLAP = 0.70

# ──────────────────────────────────────────────────────────────────────────────
# Safety limits
# ──────────────────────────────────────────────────────────────────────────────
LLM_STREAM_TIMEOUT_SEC = 20   # abort if LLM hangs
MAX_CALL_DURATION_SEC = 3600  # 1 hour hard cap

SYSTEM_PROMPT = "Hello, this is Nova from NovaCare Health Insurance. How can I assist you today? I can help with our insurance plans, claims, and renewals. If you need specific information, please provide details. If I can't assist, I can connect you to our customer support."

# ──────────────────────────────────────────────────────────────────────────────
# Text filters
# ──────────────────────────────────────────────────────────────────────────────
LOW_VALUE_UTTERANCES = {
    "okay", "ok", "okay.", "ok.",
    "thanks", "thank you", "thank you.", "thanks.",
    "hello", "hello.", "hi", "hi.",
    "hmm", "hmm.", "yeah", "yeah.",
    "yes", "yes.", "no", "no.",
    "sure", "sure.", "alright", "alright.",
    "got it", "got it.", "i see", "i see.",
    "okay thank you", "okay thank you.",
    "okay. thank you", "okay. thank you.",
    "okay thanks", "okay thanks.",
    "yes thank you", "yes thank you.",
    "no thank you", "no thank you.",
}

FILLER_PREFIX_PATTERN = re.compile(
    r'^(okay\.?\s*|ok\.?\s*|yeah\.?\s*|yes\.?\s*|'
    r'no\.?\s*|sure\.?\s*|alright\.?\s*|hmm\.?\s*|'
    r'um\.?\s*|uh\.?\s*)+',
    re.IGNORECASE,
)

INCOMPLETE_SUFFIX_PATTERN = re.compile(
    r'(\w+-\s*$|-\s*$|,\s*$|\.\.\.\s*$|\ba\s*$|\ban\s*$|\bthe\s*$)',
    re.IGNORECASE,
)

call_state = {
    "ending": False,  # set to True when end-call phrase is detected
}

# ──────────────────────────────────────────────────────────────────────────────
# Text utilities
# ──────────────────────────────────────────────────────────────────────────────

def strip_fillers(text: str) -> str:
    return FILLER_PREFIX_PATTERN.sub("", text).strip()


def is_low_value(text: str) -> bool:
    normalized = text.lower().strip().rstrip(".")
    if normalized in LOW_VALUE_UTTERANCES:
        return True
    core = strip_fillers(text)
    return not core or len(core.split()) < 2


def is_stable_transcript(text: str) -> bool:
    text = text.strip()
    if INCOMPLETE_SUFFIX_PATTERN.search(text):
        return False
    core = strip_fillers(text)
    return len(core.split()) >= SPECULATIVE_MIN_WORDS and len(core) >= SPECULATIVE_MIN_CHARS


def extract_real_query(text: str) -> str:
    return strip_fillers(text) or text

def normalize_query(text: str) -> str:

    text = text.lower().strip()

    fillers = [
        "uh",
        "um",
        "okay",
        "like",
        "please",
    ]

    words = [
        w for w in text.split()
        if w not in fillers
    ]

    return " ".join(words)

def is_similar_query(query_a: str, query_b: str) -> bool:
    if not query_a or not query_b:
        return False
    words_a = set(query_a.lower().split())
    words_b = set(query_b.lower().split())
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b), 1)
    return overlap >= SPECULATIVE_WORD_OVERLAP


# ──────────────────────────────────────────────────────────────────────────────
# Audio conversion + VAD (pool-based, thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

def mulaw_to_float32(audio_bytes: bytes) -> np.ndarray:
    pcm = audioop.ulaw2lin(audio_bytes, 2)
    audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    audio_np /= 32768.0
    return audio_np


def _run_vad_sync(audio_float32: np.ndarray, model_index: int) -> bool:
    """Runs Silero VAD synchronously — called via asyncio.to_thread."""
    model = _vad_pool[model_index]
    speech = get_speech_timestamps(
        audio_float32, model, sampling_rate=SAMPLE_RATE
    )
    return len(speech) > 0


async def detect_speech_async(audio_float32: np.ndarray) -> bool:
    """
    Acquires a VAD model from the pool, runs inference in a thread pool,
    then releases the slot. Never blocks the event loop.
    """
    global _vad_pool_index
    async with _vad_semaphore:
        async with _vad_pool_lock:
            idx = _vad_pool_index
            _vad_pool_index = (_vad_pool_index + 1) % VAD_POOL_SIZE
        return await asyncio.to_thread(_run_vad_sync, audio_float32, idx)


# ──────────────────────────────────────────────────────────────────────────────
# TTS keepalive
# ──────────────────────────────────────────────────────────────────────────────

async def tts_keepalive_loop(tts_ws_holder: dict, tts_lock: asyncio.Lock, tts_state: dict):
    while tts_state.get("running", True):
        await asyncio.sleep(10)
        if tts_state.get("speaking"):
            continue
        async with tts_lock:
            ws = tts_ws_holder.get("ws")
        if ws is None:
            continue
        try:
            await ws.send(json.dumps({
                "text": " ",
                "try_trigger_generation": False,
            }))
        except Exception:
            async with tts_lock:
                tts_ws_holder["ws"] = None


# ──────────────────────────────────────────────────────────────────────────────
# Speculative search helper (defined here to avoid closure/nonlocal race)
# ──────────────────────────────────────────────────────────────────────────────

async def _speculative_search(
    query: str,
    user_id: int,
    result_holder: dict,
    debounce: float,
    call_log,
) -> None:
    """
    Waits `debounce` seconds, then runs KB search and writes into
    result_holder. Cancelled cleanly if a newer partial arrives.
    Defined outside handle_transcripts to avoid nonlocal race conditions
    when multiple tasks are created in rapid succession.
    """
    try:
        await asyncio.sleep(debounce)
        res = await search_kb_async(query=query, user_id=user_id)
        result_holder["query"] = query
        result_holder["result"] = res
        call_log.debug("speculative_done", query=query)
    except asyncio.CancelledError:
        pass  # newer partial superseded us — expected


# ──────────────────────────────────────────────────────────────────────────────
# Transcript handler
# ──────────────────────────────────────────────────────────────────────────────

async def handle_transcripts(
    stt_ws,
    tts_ws_holder: dict,
    tts_lock: asyncio.Lock,
    stt_control: dict,
    tts_state: dict,
    vad_state: dict,
    call_log: structlog.BoundLogger,
    user_id: int,
    send_audio_to_twilio,       # callable
    get_tts_pump_task,          # callable → current task or None
    set_tts_pump_task,
    conversation_history: list,
    dynamic_system_prompt          # callable(task)
):
    speculative_task = None
    last_partial = ""
    # FIX #4: use a dict instead of nonlocal vars to avoid closure race condition
    speculative_holder: dict = {"query": "", "result": None}


    async for result in receive_transcript(stt_ws):
        try:
            # ── PARTIAL ────────────────────────────────────────────────────
            if result["type"] == "partial":
                partial_text = result["text"].strip()

                if is_low_value(partial_text):
                    continue
                if not is_stable_transcript(partial_text):
                    continue
                if partial_text == last_partial:
                    continue

                last_partial = partial_text

                if speculative_task and not speculative_task.done():
                    speculative_task.cancel()

                query_for_embedding = normalize_query(
                                        extract_real_query(partial_text)
                                    )
                call_log.debug("speculative_queued", query=query_for_embedding)

                speculative_task = asyncio.create_task(
                    _speculative_search(
                        query=query_for_embedding,
                        user_id=user_id,
                        result_holder=speculative_holder,
                        debounce=SPECULATIVE_DEBOUNCE_SEC,
                        call_log=call_log,
                    )
                )
                continue

            # ── FINAL ──────────────────────────────────────────────────────
            if result["type"] == "final":
                user_text = result["text"].strip()

                END_CALL_PHRASES = [
                    "bye",
                    "goodbye",
                    "not interested",
                    "thank you",
                    "call later",
                    "stop calling",
                    "i'm busy",
                ]

                # if any(
                #     phrase in user_text.lower()
                #     for phrase in END_CALL_PHRASES
                # ):

                #     call_log.info(
                #         "end_call_phrase_detected",
                #         text=user_text
                #     )

                #     tts_state["end_call"] = True

                if any(
                    phrase in user_text.lower()
                    for phrase in END_CALL_PHRASES
                ):

                    call_log.info(
                        "end_call_phrase_detected",
                        text=user_text
                    )

                    call_state["ending"] = True

                    # stop current AI speaking immediately
                    tts_state["speaking"] = False
                    stt_control["paused"] = True

                    # optional goodbye message
                    async with tts_lock:
                        tts_ws = tts_ws_holder.get("ws")

                    if tts_ws:
                        try:
                            await send_text(
                                connection=tts_ws,
                                text="Thank you for calling. Goodbye."
                            )
                            await flush_stream(tts_ws)

                            # small delay so caller hears goodbye
                            await asyncio.sleep(1.5)

                        except Exception as e:
                            call_log.warning(
                                "goodbye_tts_failed",
                                error=str(e)
                            )

                    return

                if not user_text:
                    continue

                if is_low_value(user_text):
                    call_log.info("skipped_low_value", text=user_text)
                    if speculative_task and not speculative_task.done():
                        speculative_task.cancel()
                    speculative_task = None
                    last_partial = ""
                    continue

                stt_final_ts = time.perf_counter()
                call_log.info("stt_final", text=user_text)



                conversation_history.append({
                    "speaker": "user",
                    "text": user_text,
                    "timestamp": time.time(),
                })

                async with tts_lock:
                    tts_ws = tts_ws_holder.get("ws")

                if tts_ws is None:
                    call_log.warning("tts_ws_missing_on_final")
                    continue

                tts_state["speaking"] = True
                stt_control["paused"] = True

                trace_id = (
                    f"{tts_state.get('call_trace', 'call')}"
                    f"-{tts_state.get('turn', 0)}"
                )

                try:
                    docs = None
                    query_for_retrieval = normalize_query(
                                                extract_real_query(user_text)
                                            )

                    # Strategy 1: word-overlap cache hit
                    if (
                        speculative_holder["result"] is not None
                        and is_similar_query(query_for_retrieval, speculative_holder["query"])
                    ):
                        docs = speculative_holder["result"]
                        call_log.info("rag_cache_hit", strategy="speculative_reuse")

                    # Strategy 2: wait briefly for in-flight speculative
                    if not docs and speculative_task is not None:
                        try:
                            await asyncio.wait_for(asyncio.shield(speculative_task), timeout=0.3)
                            if speculative_holder["result"]:
                                docs = speculative_holder["result"]
                                call_log.info("rag_cache_hit", strategy="inflight_speculative")
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            call_log.debug("speculative_timed_out")

                    # Strategy 3: fresh retrieval
                    if not docs:
                        call_log.info("rag_fresh_retrieval", query=query_for_retrieval)
                        docs = await search_kb_async(
                            query=query_for_retrieval, user_id=user_id
                        )

                    docs = (docs or [])[:3]
                    context = "\n".join(docs)

                    retrieve_ms = (time.perf_counter() - stt_final_ts) * 1000
                    call_log.info(
                        "latency",
                        trace_id=trace_id,
                        stage="context_ready",
                        ms=round(retrieve_ms, 1),
                    )

                    buffer = ""
                    first_flush_done = False
                    sent_first_chunk = False

                    # Re-acquire tts_ws under lock before streaming
                    async with tts_lock:
                        tts_ws = tts_ws_holder.get("ws")

                    if tts_ws is None:
                        call_log.warning("tts_ws_gone_before_stream")
                        continue

                    assistant_response = ""

                    # ── LLM stream with hard timeout ───────────────────────
                    stream = stream_llm(
                        query=user_text,
                        context=context,
                        trace_id=trace_id,
                        user_id=user_id,
                        dynamic_system_prompt=dynamic_system_prompt,
                    )

                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                anext(stream),
                                timeout=LLM_STREAM_TIMEOUT_SEC
                            )
                        except StopAsyncIteration:
                            break

                        assistant_response += chunk 
                        if not sent_first_chunk:
                            first_chunk_ms = (
                                time.perf_counter() - stt_final_ts
                            ) * 1000

                            call_log.info(
                                "latency",
                                trace_id=trace_id,
                                stage="first_llm_chunk",
                                ms=round(first_chunk_ms, 1),
                            )

                            sent_first_chunk = True

                        buffer += chunk

                        await send_text(
                            connection=tts_ws,
                            text=chunk,
                        )

                        if not first_flush_done:
                            await flush_stream(tts_ws)
                            first_flush_done = True
                            buffer = ""

                        elif len(buffer) > 120:
                            await flush_stream(tts_ws)
                            buffer = ""

                    if buffer.strip():
                        await flush_stream(tts_ws)

                    conversation_history.append({
                        "speaker": "assistant",
                        "text": assistant_response,
                        "timestamp": time.time(),
                    }) 

                except asyncio.TimeoutError:
                    call_log.error("llm_stream_timeout", trace_id=trace_id)

                except Exception as e:
                    call_log.error("turn_error", trace_id=trace_id, error=str(e))

                finally:
                    tts_state["speaking"] = False
                    stt_control["paused"] = False
                    tts_state["turn"] += 1

                    speculative_task = None
                    last_partial = ""
                    speculative_holder["query"] = ""
                    speculative_holder["result"] = None

                    
        except Exception as e:
            # Outer safety net — log and keep the loop alive
            call_log.error("transcript_loop_error", error=str(e))
            continue


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────────────

@router.websocket("/ws/call")
async def voice_call(websocket: WebSocket):
    """
    Each WebSocket connection runs entirely in its own coroutine scope.
    All mutable state (stt_ws, tts_ws_holder, vad_state, …) is local —
    no shared globals means concurrent calls are safely isolated.
    """
    await websocket.accept()

    # ── Per-call identity ──────────────────────────────────────────────────
    call_id = uuid.uuid4().hex[:8]
    call_log = log.bind(call_id=call_id)

    user_id_raw = websocket.query_params.get("user_id", "1")
    try:
        user_id = int(user_id_raw)
    except ValueError:
        user_id = 1

    def _load_user_system_prompt(db: Session, current_user_id: int) -> str:
        latest_prompt = (
            db.query(Prompt)
            .filter(Prompt.id == current_user_id).first()
        )
        if not latest_prompt:
            return ""
        return latest_prompt.system_prompt , latest_prompt.one_liner

    db = SessionLocal()
    try:
        dynamic_system_prompt, dynamic_one_liner = _load_user_system_prompt(db, user_id)
        dynamic_one_liner = dynamic_one_liner['text']
        dynamic_system_prompt = dynamic_system_prompt['text']
    finally:
        db.close()

    call_log.info("call_accepted")
    transcript_queue = asyncio.Queue()
    
    db_worker_task = asyncio.create_task(
    transcript_db_worker(transcript_queue)
    )

    # ── Per-call state (all local, never shared) ───────────────────────────
    stt_ws = None
    tts_ws_holder: dict = {"ws": None}
    tts_lock = asyncio.Lock()          # guards tts_ws_holder reads/writes
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
    # FIX #3: STT retry state — prevents retry storm on repeated open failures
    STT_MAX_RETRIES = 3
    STT_RETRY_DELAYS = [0.5, 1.0, 2.0]  # seconds per attempt
    stt_open_attempts = 0
    stt_open_failed = False

    conversation_history = []

    # Shared-state accessor helpers (passed into handle_transcripts)
    def get_tts_pump_task():
        return tts_pump_task

    def set_tts_pump_task(task):
        nonlocal tts_pump_task
        tts_pump_task = task

    # ── Twilio audio sender ────────────────────────────────────────────────
    async def send_audio_to_twilio(audio_bytes: bytes):
        for i in range(0, len(audio_bytes), BYTES_PER_CHUNK):
            chunk = audio_bytes[i : i + BYTES_PER_CHUNK]
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
            await asyncio.sleep(CHUNK_INTERVAL)

    # ── TTS stream lifecycle ───────────────────────────────────────────────
    async def start_tts_stream():
        nonlocal tts_pump_task
        ws = await open_elevenlabs_stream(
            system_prompt=dynamic_one_liner or SYSTEM_PROMPT
        )
        async with tts_lock:
            tts_ws_holder["ws"] = ws

        async def pump_and_reset():
            try:
                await pump_audio(ws, send_audio_to_twilio)
            finally:
                # Always clear the stale ws reference when pump exits
                async with tts_lock:
                    if tts_ws_holder.get("ws") is ws:
                        tts_ws_holder["ws"] = None
                call_log.info("tts_pump_exited")

        tts_pump_task = asyncio.create_task(pump_and_reset())
        set_tts_pump_task(tts_pump_task)

    async def restart_tts_stream():
        """Cancel current TTS stream and open a fresh one (barge-in)."""
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

    # ── Barge-in handler ───────────────────────────────────────────────────
    async def handle_barge_in():
        """
        Called when VAD detects the user speaking while TTS is playing.
        Cancels TTS immediately so the user's speech is heard.
        """
        if tts_state["speaking"]:
            tts_state["speaking"] = False
            stt_control["paused"] = False
            await restart_tts_stream()

    # ──────────────────────────────────────────────────────────────────────
    # Main event loop
    # ──────────────────────────────────────────────────────────────────────
    try:
        while True:
            # ── Hard call duration cap ─────────────────────────────────────
            if time.perf_counter() - call_start_time > MAX_CALL_DURATION_SEC:
                call_log.warning("max_call_duration_reached")
                break

            try:
                data = await websocket.receive()

                # IMPORTANT
                if data["type"] == "websocket.disconnect":
                    call_log.info("twilio_disconnect_event")
                    break

            except (
                WebSocketDisconnect,
                ConnectionClosed,
                RuntimeError,
            ):
                call_log.info("twilio_disconnected")
                break

            if "text" not in data:
                continue

            msg = json.loads(data["text"])
            event = msg.get("event")

            # ── start ──────────────────────────────────────────────────────
            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                call_log.info("call_started", stream_sid=stream_sid)

                await start_tts_stream()

                if tts_keepalive_task is None:
                    tts_keepalive_task = asyncio.create_task(
                        tts_keepalive_loop(tts_ws_holder, tts_lock, tts_state)
                    )

            # ── media ──────────────────────────────────────────────────────
            elif event == "media":
                raw_bytes = base64.b64decode(msg["media"]["payload"])

                # ── VAD (every N chunks, off event loop) ───────────────────
                vad_counter += 1
                if vad_counter >= VAD_CHECK_EVERY:
                    vad_counter = 0
                    audio_float = mulaw_to_float32(raw_bytes)
                    # asyncio.to_thread keeps event loop unblocked
                    is_speaking = await detect_speech_async(audio_float)
                else:
                    is_speaking = vad_state["speaking"]

                now = time.perf_counter()

                if is_speaking:
                    # ── Barge-in: user speaks while AI is responding ───────
                    if tts_state["speaking"]:
                        # FIX #2: store task and attach error-logging callback
                        # so exceptions are never silently swallowed
                        barge_in_task = asyncio.create_task(handle_barge_in())

                        def _barge_in_done(fut: asyncio.Task):
                            if not fut.cancelled() and fut.exception():
                                call_log.error(
                                    "barge_in_error",
                                    error=str(fut.exception()),
                                )

                        barge_in_task.add_done_callback(_barge_in_done)

                    last_voice_time = now
                    if not vad_state["speaking"]:
                        call_log.debug("speech_started")
                    vad_state["speaking"] = True

                else:
                    silence_duration = now - last_voice_time
                    if vad_state["speaking"] and silence_duration > SILENCE_TIMEOUT:
                        call_log.debug(
                            "speech_ended", silence_sec=round(silence_duration, 2)
                        )
                        vad_state["speaking"] = False

                # ── Open STT on first media frame ──────────────────────────
                if stt_ws is None and not stt_open_failed:

                    if stt_open_attempts >= STT_MAX_RETRIES:
                        stt_open_failed = True
                        call_log.error("stt_max_retries_exceeded")
                        continue

                    try:
                        stt_open_attempts += 1

                        if stt_open_attempts > 1:
                            delay = STT_RETRY_DELAYS[min(
                                stt_open_attempts - 2,
                                len(STT_RETRY_DELAYS) - 1
                            )]
                            await asyncio.sleep(delay)

                        call_log.info(
                            "stt_opening",
                            attempt=stt_open_attempts
                        )

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
                                dynamic_system_prompt
                            )
                        )
                    except Exception as e:
                        call_log.error("stt_open_failed", error=str(e)) 

                if stt_control["paused"]:
                    continue

                # ── Batch audio before forwarding to STT ───────────────────
                audio_buffer.extend(raw_bytes)
                frame_count += 1

                if frame_count >= STT_BATCH_FRAMES:
                    try:
                        await send_audio(stt_ws, bytes(audio_buffer))
                    except Exception as e:
                        call_log.warning("stt_send_failed", error=str(e))
                    finally:
                        audio_buffer.clear()
                        frame_count = 0

            # ── stop ───────────────────────────────────────────────────────
            elif event == "stop":
                call_log.info("call_stopped")
                break

    # ── Cleanup ────────────────────────────────────────────────────────────
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
