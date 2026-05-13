# text_utils.py

## Purpose
Text cleanup and query normalization utilities used before retrieval.

## Main functions
- `strip_fillers(text)`
- `is_low_value(text)`
- `is_stable_transcript(text)`
- `extract_real_query(text)`
- `normalize_query(text)`
- `is_similar_query(query_a, query_b)`

## Used by
- `transcript_flow.py`

## Edit here when
- You want better filler handling, stability checks, or similarity matching.
