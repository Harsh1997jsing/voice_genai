import asyncio
import audioop
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps

from app.services.calling.constants import SAMPLE_RATE, VAD_POOL_SIZE

_vad_pool: list = [load_silero_vad() for _ in range(VAD_POOL_SIZE)]
_vad_semaphore = asyncio.Semaphore(VAD_POOL_SIZE)
_vad_pool_lock = asyncio.Lock()
_vad_pool_index = 0


def mulaw_to_float32(audio_bytes: bytes) -> np.ndarray:
    pcm = audioop.ulaw2lin(audio_bytes, 2)
    audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    audio_np /= 32768.0
    return audio_np


def _run_vad_sync(audio_float32: np.ndarray, model_index: int) -> bool:
    model = _vad_pool[model_index]
    speech = get_speech_timestamps(audio_float32, model, sampling_rate=SAMPLE_RATE)
    return len(speech) > 0


async def detect_speech_async(audio_float32: np.ndarray) -> bool:
    global _vad_pool_index
    async with _vad_semaphore:
        async with _vad_pool_lock:
            idx = _vad_pool_index
            _vad_pool_index = (_vad_pool_index + 1) % VAD_POOL_SIZE
        return await asyncio.to_thread(_run_vad_sync, audio_float32, idx)
