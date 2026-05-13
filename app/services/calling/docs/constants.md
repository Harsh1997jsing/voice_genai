# constants.py

## Purpose
Central config and shared constants for the call runtime.

## Contains
- Audio constants (`SAMPLE_RATE`, `CHUNK_MS`, `BYTES_PER_CHUNK`, `STT_BATCH_FRAMES`)
- VAD tuning (`VAD_POOL_SIZE`, `VAD_CHECK_EVERY`, `SILENCE_TIMEOUT`)
- Speculative retrieval tuning (`SPECULATIVE_*`)
- Safety limits (`LLM_STREAM_TIMEOUT_SEC`, `MAX_CALL_DURATION_SEC`)
- Default assistant prompt (`SYSTEM_PROMPT`)
- Text filtering patterns and utterance sets
- Shared call flag: `call_state`

## Edit here when
- You want to tune latency, VAD frequency, timeout, or default prompt without changing runtime logic.
