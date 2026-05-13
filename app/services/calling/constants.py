import re

SAMPLE_RATE = 8000
CHUNK_MS = 20
BYTES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_MS / 1000)
CHUNK_INTERVAL = CHUNK_MS / 1000
STT_BATCH_FRAMES = 2

VAD_POOL_SIZE = 8
VAD_CHECK_EVERY = 5
SILENCE_TIMEOUT = 0.7

SPECULATIVE_MIN_WORDS = 4
SPECULATIVE_MIN_CHARS = 20
SPECULATIVE_DEBOUNCE_SEC = 0.5
SPECULATIVE_WORD_OVERLAP = 0.70

LLM_STREAM_TIMEOUT_SEC = 20
MAX_CALL_DURATION_SEC = 3600

SYSTEM_PROMPT = (
    "Hello, this is Nova from NovaCare Health Insurance. How can I assist you today? "
    "I can help with our insurance plans, claims, and renewals. If you need specific "
    "information, please provide details. If I can't assist, I can connect you to our "
    "customer support."
)

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

call_state = {"ending": False}
