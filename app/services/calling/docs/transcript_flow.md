# transcript_flow.py

## Purpose
Core transcript turn pipeline (partial/final STT -> RAG -> LLM -> TTS text stream).

## Main functions
- `tts_keepalive_loop(...)`
- `_speculative_search(...)`
- `handle_transcripts(...)`

## Responsibilities
- Filters partial/final transcript noise
- Runs speculative retrieval and cache reuse
- Runs fresh retrieval fallback
- Streams LLM response and sends chunks to TTS
- Manages per-turn state reset and error handling
- Handles end-call phrases and goodbye behavior

## Depends on
- `text_utils.py`, `constants.py`
- STT/TTS services, KB service, RAG pipeline
