# Calling Service Guide

This folder contains the full voice-calling runtime used by the `/ws/call` WebSocket route.

## High-level flow

1. `app/api/calling_router.py` receives WebSocket connection.
2. Router calls `handle_voice_call()` from `app/services/calling`.
3. `session.py` manages call lifecycle (start/media/stop, STT/TTS setup, cleanup).
4. `transcript_flow.py` handles transcript events (partial/final), retrieval, LLM response, and TTS text streaming.
5. `vad_audio.py` detects user speech for barge-in behavior.
6. `text_utils.py` filters and normalizes transcript text.

---

## File-by-file explanation

### `__init__.py`

Purpose:
- Public export for the package.

What it does:
- Re-exports `handle_voice_call` so imports can stay simple:
  - `from app.services.calling import handle_voice_call`

---

### `constants.py`

Purpose:
- Central place for tunables and shared static values.

Main content:
- Audio settings: `SAMPLE_RATE`, `CHUNK_MS`, `BYTES_PER_CHUNK`, `STT_BATCH_FRAMES`
- VAD settings: `VAD_POOL_SIZE`, `VAD_CHECK_EVERY`, `SILENCE_TIMEOUT`
- Speculative retrieval settings:
  - `SPECULATIVE_MIN_WORDS`
  - `SPECULATIVE_MIN_CHARS`
  - `SPECULATIVE_DEBOUNCE_SEC`
  - `SPECULATIVE_WORD_OVERLAP`
- Safety limits: `LLM_STREAM_TIMEOUT_SEC`, `MAX_CALL_DURATION_SEC`
- Default agent greeting prompt: `SYSTEM_PROMPT`
- Text filters and regex patterns for filler/incomplete transcript detection
- Call-level flag dictionary: `call_state`

When to edit this file:
- You want to tune latency/quality behavior without touching logic.

---

### `text_utils.py`

Purpose:
- Transcript text cleanup and query similarity logic.

Main functions:
- `strip_fillers(text)`: Removes filler prefixes (`okay`, `uh`, etc.).
- `is_low_value(text)`: Detects low-information utterances.
- `is_stable_transcript(text)`: Checks if partial text is stable enough for speculative retrieval.
- `extract_real_query(text)`: Returns cleaned query text.
- `normalize_query(text)`: Light normalization before KB search.
- `is_similar_query(a, b)`: Overlap-based similarity used to reuse speculative retrieval result.

When to edit this file:
- You want better multilingual filler removal or smarter query matching.

---

### `vad_audio.py`

Purpose:
- Audio decoding and speech activity detection.

Main functions:
- `mulaw_to_float32(audio_bytes)`: Converts Twilio mu-law bytes to float32 PCM for VAD.
- `detect_speech_async(audio_float32)`: Async speech detection using pooled Silero models.

Key design:
- Uses a model pool + semaphore + lock to avoid shared-model race/corruption under concurrent calls.
- Offloads VAD inference via `asyncio.to_thread` to keep event loop responsive.

When to edit this file:
- You change VAD engine/model or audio conversion behavior.

---

### `transcript_flow.py`

Purpose:
- Handles STT transcript stream and creates assistant responses.

Main responsibilities:
- Keepalive ping for TTS websocket (`tts_keepalive_loop`)
- Speculative retrieval scheduling (`_speculative_search`)
- Main transcript handler (`handle_transcripts`) for:
  - partial transcript filtering
  - speculative retrieval
  - final transcript handling
  - low-value skip
  - end-call phrase handling
  - KB retrieval strategy (reuse speculative -> wait in-flight -> fresh retrieval)
  - LLM streaming (`stream_llm`)
  - sending chunks to TTS websocket
  - turn state reset and error handling

Inputs to `handle_transcripts`:
- `stt_ws`, `tts_ws_holder`, `tts_lock`, `stt_control`, `tts_state`, `vad_state`
- `call_log`, `user_id`, callback helpers, `conversation_history`, `dynamic_system_prompt`

Outputs/side effects:
- Streams text to TTS
- Updates call states
- Appends transcript entries to `conversation_history`

When to edit this file:
- You want to change answer behavior, retrieval strategy, or turn orchestration.

---

### `session.py`

Purpose:
- Orchestrates full WebSocket call session lifecycle.

Main responsibilities:
- Accept websocket and create per-call state
- Read `user_id` from query params
- Load user prompt/one-liner from DB (`_load_user_prompts`)
- Manage Twilio events:
  - `start`: open TTS stream and keepalive task
  - `media`: run VAD + STT open/send pipeline + barge-in restart logic
  - `stop`: terminate call loop
- Start `handle_transcripts` background task once STT opens
- Persist transcript on cleanup (`save_call_transcript_to_db_sync`)
- Cancel/close all child tasks and sockets safely in `finally`

Important nested helpers in this file:
- `send_audio_to_twilio`
- `start_tts_stream`
- `restart_tts_stream`
- `handle_barge_in`

When to edit this file:
- You want to change lifecycle behavior, event loop flow, or resource cleanup logic.

---

## Where to change common requirements

- Change default assistant intro prompt:
  - `constants.py` -> `SYSTEM_PROMPT`
- Tune speculative retrieval aggressiveness:
  - `constants.py` + `text_utils.py`
- Change end-call phrases:
  - `transcript_flow.py` (`end_call_phrases` list)
- Change VAD sensitivity/interval:
  - `constants.py` and potentially `vad_audio.py`
- Change call timeout:
  - `constants.py` -> `MAX_CALL_DURATION_SEC`

---

## Dependency map

- `session.py` depends on:
  - `constants.py`
  - `vad_audio.py`
  - `transcript_flow.py`
  - shared STT/TTS/DB services
- `transcript_flow.py` depends on:
  - `text_utils.py`
  - `constants.py`
  - KB/RAG/STT/TTS services
- `text_utils.py` depends on:
  - `constants.py`
- `__init__.py` exposes `handle_voice_call` from `session.py`
