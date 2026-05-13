# vad_audio.py

## Purpose
Audio decoding + speech activity detection (VAD).

## Main functions
- `mulaw_to_float32(audio_bytes)` converts Twilio mu-law audio to float32 PCM.
- `detect_speech_async(audio_float32)` runs Silero VAD asynchronously.

## Key behavior
- Uses VAD model pool with semaphore/lock for concurrency safety.
- Runs inference with `asyncio.to_thread` to avoid event-loop blocking.

## Used by
- `session.py`
