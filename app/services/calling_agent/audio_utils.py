"""
audio_utils.py — Audio format conversion between Twilio (mulaw 8kHz) and
ElevenLabs Agent (pcm_16000 by default).

─────────────────────────────────────────────────────────────────────────────
IMPORTANT — Read this before using:

Twilio Media Streams send/receive:  mulaw, 8 kHz, mono
ElevenLabs Agent default output:    PCM 16-bit, 16 kHz, mono

You need conversion in BOTH directions unless you configure the ElevenLabs
Agent to use 'ulaw_8000' output format.

RECOMMENDED: In the ElevenLabs dashboard →
  Agent Settings → Advanced → Output Audio Format → ulaw_8000

If you do that, set AGENT_OUTPUT_IS_MULAW = True below and skip conversion
on the output side entirely. You only need to convert input if required.

─────────────────────────────────────────────────────────────────────────────
This module provides two conversion paths:

  1. mulaw_to_pcm16(mulaw_bytes)  → PCM16 16kHz bytes  (Twilio → Agent input)
  2. pcm16_to_mulaw(pcm16_bytes)  → mulaw 8kHz bytes   (Agent output → Twilio)

Both use numpy (already in your project for VAD). No audioop dependency
(removed in Python 3.13).
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np

# ─── Set this True if your ElevenLabs Agent outputs ulaw_8000 ────────────────
# Then: pcm16_to_mulaw() is never called; agent audio goes straight to Twilio.
AGENT_OUTPUT_IS_MULAW: bool = False   # ← change to True after dashboard config


# ──────────────────────────────────────────────────────────────────────────────
# μ-law encode / decode tables (ITU-T G.711)
# ──────────────────────────────────────────────────────────────────────────────

_MULAW_BIAS   = 0x84   # 132
_MULAW_MAX    = 32767
_MULAW_CLIP   = 32635

def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """
    Decode mulaw 8kHz → PCM16 16kHz (upsample 2×).

    Input:  bytes,  length N  (one sample per byte, mulaw encoded)
    Output: bytes,  length 4N (two int16 samples per input sample, little-endian)
    """
    # ── 1. mulaw → linear int16 (8kHz) ───────────────────────────────────────
    samples_8k = _mulaw_decode(np.frombuffer(mulaw_bytes, dtype=np.uint8))

    # ── 2. Upsample 8kHz → 16kHz (linear interpolation) ─────────────────────
    samples_16k = _upsample_2x(samples_8k)

    return samples_16k.astype("<i2").tobytes()


def pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """
    Encode PCM16 16kHz → mulaw 8kHz (downsample 2×).

    Input:  bytes, length 2N (int16 little-endian, 16kHz)
    Output: bytes, length N  (mulaw 8kHz, one byte per sample)
    """
    samples_16k = np.frombuffer(pcm16_bytes, dtype="<i2")

    # ── 1. Downsample 16kHz → 8kHz (average pairs) ────────────────────────────
    samples_8k = _downsample_2x(samples_16k)

    # ── 2. linear int16 → mulaw ────────────────────────────────────────────────
    return _mulaw_encode(samples_8k).tobytes()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mulaw_decode(mulaw: np.ndarray) -> np.ndarray:
    """ITU-T G.711 μ-law decode: uint8 → int16."""
    mulaw  = ~mulaw.astype(np.int32)
    sign   = mulaw & 0x80
    exp    = (mulaw >> 4) & 0x07
    mant   = mulaw & 0x0F
    linear = ((mant << 1) + 33) << (exp + 2)
    linear = np.where(sign != 0, -linear, linear)
    return np.clip(linear, -32768, 32767).astype(np.int16)


def _mulaw_encode(linear: np.ndarray) -> np.ndarray:
    """ITU-T G.711 μ-law encode: int16 → uint8."""
    lin   = linear.astype(np.int32)
    sign  = (lin >> 8) & 0x80
    lin   = np.abs(lin)
    lin   = np.clip(lin + _MULAW_BIAS, 0, _MULAW_MAX)

    exp   = np.zeros_like(lin)
    for bit in range(7, 0, -1):
        mask        = lin >= (1 << (bit + 3))
        exp[mask]   = bit

    mant  = (lin >> (exp + 3)) & 0x0F
    mulaw = ~(sign | (exp << 4) | mant)
    return mulaw.astype(np.uint8)


def _upsample_2x(samples: np.ndarray) -> np.ndarray:
    """Simple 2× upsample via linear interpolation."""
    out           = np.empty(len(samples) * 2, dtype=np.int32)
    out[0::2]     = samples
    out[1:-1:2]   = (samples[:-1].astype(np.int32) + samples[1:].astype(np.int32)) // 2
    out[-1]       = samples[-1]
    return np.clip(out, -32768, 32767).astype(np.int16)


def _downsample_2x(samples: np.ndarray) -> np.ndarray:
    """Simple 2× downsample by averaging adjacent pairs."""
    if len(samples) % 2:
        samples = np.append(samples, samples[-1])
    pairs = samples.reshape(-1, 2).astype(np.int32)
    return np.clip(pairs.mean(axis=1), -32768, 32767).astype(np.int16)