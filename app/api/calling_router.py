import asyncio
import json
import base64
import time
import uuid
import re
from contextlib import suppress
import audioop
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from silero_vad import load_silero_vad, get_speech_timestamps

from app.services.stt_service import (
    open_stt,
    send_audio,
    receive_transcript,
)

from app.services.tts_service import (
    open_elevenlabs_stream,
    send_text,
    pump_audio,
    flush_stream,
)

from app.services.kb_service import search_kb_async
from app.services.rag_pipeline import stream_llm


router = APIRouter()

# =========================================================
# AUDIO SETTINGS
# =========================================================

SAMPLE_RATE = 8000
CHUNK_MS = 20

BYTES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_INTERVAL = CHUNK_MS / 1000

STT_BATCH_FRAMES = 2          # ✅ reduced STT websocket spam (was 1)

SYSTEM_PROMPT = "I am NovaCare Health Insurance AI assistant."

# =========================================================
# SPECULATIVE RETRIEVAL SETTINGS
# =========================================================

SPECULATIVE_MIN_WORDS = 4
SPECULATIVE_MIN_CHARS = 20
SPECULATIVE_DEBOUNCE_SEC = 0.5   # ✅ slightly longer debounce (was 0.4)
SPECULATIVE_WORD_OVERLAP = 0.70  # ✅ word overlap threshold (replaces SequenceMatcher)

# =========================================================
# VAD SETTINGS
# =========================================================

SILENCE_TIMEOUT = 0.7
VAD_CHECK_EVERY = 5              # ✅ run Silero every 5 chunks, not every chunk

vad_model = load_silero_vad()

# =========================================================
# RETRIEVAL FILTERS
# =========================================================

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
    re.IGNORECASE
)

INCOMPLETE_SUFFIX_PATTERN = re.compile(
    r'(\w+-\s*$|-\s*$|,\s*$|\.\.\.\s*$|\ba\s*$|\ban\s*$|\bthe\s*$)',
    re.IGNORECASE
)


def strip_fillers(text: str) -> str:
    return FILLER_PREFIX_PATTERN.sub("", text).strip()


def is_low_value(text: str) -> bool:
    normalized = text.lower().strip().rstrip(".")
    if normalized in LOW_VALUE_UTTERANCES:
        return True
    core = strip_fillers(text)
    if not core or len(core.split()) < 2:
        return True
    return False


def is_stable_transcript(text: str) -> bool:
    text = text.strip()
    if INCOMPLETE_SUFFIX_PATTERN.search(text):
        return False
    core = strip_fillers(text)
    if len(core.split()) < SPECULATIVE_MIN_WORDS:
        return False
    if len(core) < SPECULATIVE_MIN_CHARS:
        return False
    return True


def extract_real_query(text: str) -> str:
    return strip_fillers(text) or text


def is_similar_query(query_a: str, query_b: str) -> bool:
    """
    ✅ Fast word-overlap check — replaces slow SequenceMatcher.
    Returns True if >70% of words overlap between the two queries.
    Handles cases like:
      speculative: 'give me insurance plan details'
      final:       'can you give me insurance plan details'
    """
    if not query_a or not query_b:
        return False
    words_a = set(query_a.lower().split())
    words_b = set(query_b.lower().split())
    overlap = len(words_a & words_b) / max(len(words_a), len(words_b), 1)
    return overlap >= SPECULATIVE_WORD_OVERLAP


# =========================================================
# AUDIO HELPERS
# =========================================================

def mulaw_to_float32(audio_bytes: bytes):
    pcm = audioop.ulaw2lin(audio_bytes, 2)
    audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    audio_np /= 32768.0
    return audio_np


def detect_speech(audio_float32):
    speech = get_speech_timestamps(
        audio_float32,
        vad_model,
        sampling_rate=SAMPLE_RATE
    )
    return len(speech) > 0


# =========================================================
# TTS KEEPALIVE
# =========================================================

async def tts_keepalive_loop(tts_ws_holder: dict, tts_state: dict):
    while tts_state.get("running", True):
        await asyncio.sleep(10)
        if tts_state.get("speaking"):
            continue
        ws = tts_ws_holder.get("ws")
        if ws is None:
            continue
        try:
            await ws.send(json.dumps({
                "text": " ",
                "try_trigger_generation": False,
            }))
        except Exception:
            tts_ws_holder["ws"] = None


# =========================================================
# TRANSCRIPT HANDLER
# =========================================================

async def handle_transcripts(
    stt_ws,
    tts_ws_holder: dict,
    stt_control: dict,
    tts_state: dict,
    vad_state: dict,           # ✅ shared VAD state from media loop
):
    speculative_task = None
    last_partial = ""
    last_speculative_query = ""
    last_speculative_result = None

    async for result in receive_transcript(stt_ws):

        # =================================================
        # PARTIAL TRANSCRIPT
        # =================================================

        if result["type"] == "partial":

            partial_text = result["text"].strip()

            # Filter 1: low-value phrases
            if is_low_value(partial_text):
                continue

            # Filter 2: unstable / incomplete transcript
            if not is_stable_transcript(partial_text):
                continue

            # Filter 3: no change
            if partial_text == last_partial:
                continue

            # ✅ Filter 4: user is still speaking — skip speculative entirely
            # No point embedding mid-sentence partials, they will be cancelled
            # if vad_state["speaking"]:
            #     continue

            last_partial = partial_text

            # Cancel previous speculative task
            if speculative_task and not speculative_task.done():
                speculative_task.cancel()

            query_for_embedding = extract_real_query(partial_text)

            print(f"[RAG] Speculative queued: {query_for_embedding!r}")

            async def debounced_speculative(query: str):
                try:
                    await asyncio.sleep(SPECULATIVE_DEBOUNCE_SEC)
                    nonlocal last_speculative_query
                    nonlocal last_speculative_result
                    last_speculative_query = query
                    res = await search_kb_async(query=query, user_id=1)
                    last_speculative_result = res
                    print(f"[RAG] Speculative done: {query!r}")
                    return res
                except asyncio.CancelledError:
                    return None

            speculative_task = asyncio.create_task(
                debounced_speculative(query_for_embedding)
            )

            continue

        # =================================================
        # FINAL TRANSCRIPT
        # =================================================

        if result["type"] == "final":

            user_text = result["text"].strip()

            if not user_text:
                continue

            # Skip low-value finals entirely — no Pinecone, no LLM
            if is_low_value(user_text):
                print(f"[RAG] Skipped low-value final: {user_text!r}")
                if speculative_task and not speculative_task.done():
                    speculative_task.cancel()
                speculative_task = None
                last_partial = ""
                continue

            stt_final_ts = time.perf_counter()
            print(f"[STT] Final: {user_text}")

            tts_ws = tts_ws_holder.get("ws")
            if tts_ws is None:
                continue

            tts_state["speaking"] = True
            stt_control["paused"] = True

            trace_id = (
                f"{tts_state.get('call_trace', 'call')}"
                f"-{tts_state.get('turn', 0)}"
            )

            try:

                # =========================================
                # SMART CONTEXT: 3-strategy retrieval
                # =========================================

                docs = None
                query_for_retrieval = extract_real_query(user_text)

                # Strategy 1: reuse cached speculative if word overlap > 70%
                if (
                    last_speculative_result is not None
                    and is_similar_query(
                        query_for_retrieval,
                        last_speculative_query
                    )
                ):
                    docs = last_speculative_result
                    print(f"[RAG] Reused speculative cache ✓")

                # Strategy 2: wait briefly for in-flight speculative task
                if not docs and speculative_task is not None:
                    try:
                        docs = await asyncio.wait_for(
                            speculative_task, timeout=0.3
                        )
                        if docs:
                            print("[RAG] Used in-flight speculative retrieval")
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        print("[RAG] Speculative timed out / cancelled")

                # Strategy 3: fresh retrieval — only if both above failed
                if not docs:
                    print(f"[RAG] Fresh retrieval: {query_for_retrieval!r}")
                    docs = await search_kb_async(
                        query=query_for_retrieval,
                        user_id=1
                    )

                if asyncio.iscoroutine(docs):
                    docs = await docs
                docs = docs[:2]
                context = "\n".join(docs)

                retrieve_ms = (time.perf_counter() - stt_final_ts) * 1000
                print(
                    f"[LATENCY] trace={trace_id} "
                    f"stage=context_ready ms={retrieve_ms:.1f}"
                )

                # =========================================
                # STREAM LLM
                # =========================================

                buffer = ""
                first_flush_done = False
                sent_first_chunk = False

                async for chunk in stream_llm(query=user_text, context=context, trace_id=trace_id):
                    buffer += chunk

                    if not sent_first_chunk:
                        first_chunk_ms = (time.perf_counter() - stt_final_ts) * 1000
                        print(f"[LATENCY] trace={trace_id} stage=first_llm_chunk ms={first_chunk_ms:.1f}")
                        sent_first_chunk = True

                    await send_text(connection=tts_ws, text=chunk)

                    # First token → flush immediately so TTS starts NOW
                    if not first_flush_done:
                        await flush_stream(tts_ws)
                        first_flush_done = True
                        buffer = ""

                    # Subsequent tokens → batch to 120 chars
                    elif len(buffer) > 120:
                        await flush_stream(tts_ws)
                        buffer = ""

                if buffer.strip():
                    await flush_stream(tts_ws)

            except Exception as e:
                print(f"[RAG/TTS] Error: {e}")

            finally:
                tts_state["speaking"] = False
                stt_control["paused"] = False
                tts_state["turn"] += 1

                # Reset all speculative state for next turn
                speculative_task = None
                last_partial = ""
                last_speculative_query = ""
                last_speculative_result = None


# =========================================================
# MAIN WEBSOCKET
# =========================================================

@router.websocket("/ws/call")
async def voice_call(websocket: WebSocket):

    print("WebSocket connection initiated")

    await websocket.accept()

    stt_ws = None
    tts_ws_holder = {"ws": None}
    stt_control = {"paused": False}
    tts_state = {
        "speaking": False,
        "running": True,
        "call_trace": f"call-{uuid.uuid4().hex[:8]}",
        "turn": 1,
    }

    # ✅ shared VAD state dict — passed into handle_transcripts
    vad_state = {"speaking": False}

    stt_task = None
    tts_keepalive_task = None
    tts_pump_task = None
    stream_sid = None
    audio_buffer = bytearray()
    frame_count = 0

    # ✅ VAD throttle counter — run Silero every 5 chunks, not every 20ms
    vad_counter = 0

    last_voice_time = time.perf_counter()

    # =====================================================
    # SEND AUDIO TO TWILIO
    # =====================================================

    async def send_audio_to_twilio(audio_bytes: bytes):
        for i in range(0, len(audio_bytes), BYTES_PER_CHUNK):
            chunk = audio_bytes[i:i + BYTES_PER_CHUNK]
            payload = base64.b64encode(chunk).decode("utf-8")
            try:
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload}
                }))
            except Exception as e:
                print(f"[TTS] Twilio send failed: {e}")
                return
            await asyncio.sleep(CHUNK_INTERVAL)

    # =====================================================
    # START TTS
    # =====================================================

    async def start_tts_stream():
        nonlocal tts_pump_task
        ws = await open_elevenlabs_stream(system_prompt=SYSTEM_PROMPT)
        tts_ws_holder["ws"] = ws
        tts_pump_task = asyncio.create_task(
            pump_audio(ws, send_audio_to_twilio)
        )

    # =====================================================
    # MAIN LOOP
    # =====================================================

    try:
        while True:
            try:
                data = await websocket.receive()
            except (WebSocketDisconnect, ConnectionClosed):
                print("Twilio disconnected")
                break

            if "text" not in data:
                continue

            msg = json.loads(data["text"])
            event = msg.get("event")

            # =============================================
            # START
            # =============================================

            if event == "start":
                stream_sid = msg["start"]["streamSid"]
                print(f"Call started: {stream_sid}")
                await start_tts_stream()
                if tts_keepalive_task is None:
                    tts_keepalive_task = asyncio.create_task(
                        tts_keepalive_loop(tts_ws_holder, tts_state)
                    )

            # =============================================
            # MEDIA
            # =============================================

            elif event == "media":
                raw_bytes = base64.b64decode(msg["media"]["payload"])

                # ✅ VAD: only run Silero every VAD_CHECK_EVERY chunks
                # reduces CPU by ~80% vs running on every 20ms chunk
                vad_counter += 1

                if vad_counter >= VAD_CHECK_EVERY:
                    vad_counter = 0
                    audio_float = mulaw_to_float32(raw_bytes)
                    is_speaking = detect_speech(audio_float)
                else:
                    is_speaking = vad_state["speaking"]

                now = time.perf_counter()

                if is_speaking:
                    last_voice_time = now
                    if not vad_state["speaking"]:
                        print("[VAD] Speech started")
                    vad_state["speaking"] = True   # ✅ update shared dict

                else:
                    silence_duration = now - last_voice_time
                    if (
                        vad_state["speaking"]
                        and silence_duration > SILENCE_TIMEOUT
                    ):
                        print(
                            f"[VAD] Speech ended "
                            f"after {silence_duration:.2f}s silence"
                        )
                        vad_state["speaking"] = False  # ✅ update shared dict

                # Open STT on first media event
                if stt_ws is None:
                    print("[STT] Opening connection")
                    stt_ws = await open_stt()
                    stt_task = asyncio.create_task(
                        handle_transcripts(
                            stt_ws,
                            tts_ws_holder,
                            stt_control,
                            tts_state,
                            vad_state,    # ✅ pass shared VAD state
                        )
                    )

                if stt_control["paused"]:
                    continue

                # ✅ Batch audio frames before sending to STT
                # reduces WebSocket calls by 3x (was STT_BATCH_FRAMES=1)
                audio_buffer.extend(raw_bytes)
                frame_count += 1

                if frame_count >= STT_BATCH_FRAMES:
                    try:
                        await send_audio(stt_ws, bytes(audio_buffer))
                    except Exception as e:
                        print(f"[STT] send failed: {e}")
                    finally:
                        audio_buffer.clear()
                        frame_count = 0

            # =============================================
            # STOP
            # =============================================

            elif event == "stop":
                print("Call stopped")
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
