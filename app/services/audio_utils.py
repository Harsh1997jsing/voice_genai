# audio_utils 

import audioop
import numpy as np

def mulaw_to_float32(audio_bytes: bytes) -> np.ndarray:
    pcm = audioop.ulaw2lin(audio_bytes, 2)

    audio_np = np.frombuffer(
        pcm,
        dtype=np.int16
    ).astype(np.float32)

    audio_np /= 32768.0

    return audio_np