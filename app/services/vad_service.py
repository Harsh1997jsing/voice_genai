import torch
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()


def detect_speech(audio_float32, sample_rate=8000):
    speech = get_speech_timestamps(
        audio_float32,
        model,
        sampling_rate=sample_rate
    )

    return len(speech) > 0


import asyncio

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
)

from app.core.constants import SAMPLE_RATE, VAD_POOL_SIZE

_vad_pool = [
    load_silero_vad()
    for _ in range(VAD_POOL_SIZE)
]

_vad_semaphore = asyncio.Semaphore(VAD_POOL_SIZE)
_vad_pool_lock = asyncio.Lock()

_vad_pool_index = 0

def _run_vad_sync(audio_float32, model_index):

    model = _vad_pool[model_index]

    speech = get_speech_timestamps(
        audio_float32,
        model,
        sampling_rate=SAMPLE_RATE
    )

    return len(speech) > 0

async def detect_speech_async(audio_float32):

    global _vad_pool_index

    async with _vad_semaphore:

        async with _vad_pool_lock:

            idx = _vad_pool_index

            _vad_pool_index = (
                _vad_pool_index + 1
            ) % VAD_POOL_SIZE

        return await asyncio.to_thread(
            _run_vad_sync,
            audio_float32,
            idx
        )