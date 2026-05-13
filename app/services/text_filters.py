import re

LOW_VALUE_UTTERANCES = {
    "okay", "ok", "thanks", "thank you",
}

FILLER_PREFIX_PATTERN = re.compile(
    r'^(okay\.?\s*|ok\.?\s*)+',
    re.IGNORECASE,
)

INCOMPLETE_SUFFIX_PATTERN = re.compile(
    r'(\w+-\s*$|,\s*$)',
    re.IGNORECASE,
)

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

    return len(core.split()) >= 4 and len(core) >= 20

def extract_real_query(text: str) -> str:
    return strip_fillers(text) or text

def normalize_query(text: str) -> str:

    fillers = ["uh", "um", "okay"]

    words = [
        w for w in text.lower().split()
        if w not in fillers
    ]

    return " ".join(words)

def is_similar_query(query_a: str, query_b: str) -> bool:

    if not query_a or not query_b:
        return False

    words_a = set(query_a.lower().split())
    words_b = set(query_b.lower().split())

    overlap = len(words_a & words_b) / max(
        len(words_a),
        len(words_b),
        1
    )

    return overlap >= 0.70