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