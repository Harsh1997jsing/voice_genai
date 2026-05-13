# session.py

## Purpose
WebSocket call session orchestrator.

## Main entrypoint
- `handle_voice_call(websocket)`

## Responsibilities
- Accepts websocket and sets per-call state
- Loads user prompt from DB
- Handles Twilio events (`start`, `media`, `stop`)
- Opens STT and starts transcript handler task
- Runs VAD + barge-in restart logic
- Sends audio back to Twilio
- Cleans up tasks/sockets and saves transcript

## Depends on
- `vad_audio.py`, `transcript_flow.py`, `constants.py`
- STT/TTS/transcript DB services
