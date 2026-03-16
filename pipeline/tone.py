from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass
class ToneEstimate:
    tone: str
    confidence: float
    contour_hz: list[float]


def estimate_f0_contour(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    import librosa

    f0, _, _ = librosa.pyin(
        audio,
        fmin=config.TONE_F0_MIN_HZ,
        fmax=config.TONE_F0_MAX_HZ,
        sr=sr,
        frame_length=2048,
        hop_length=256,
    )
    return f0


def classify_tone(f0: np.ndarray) -> ToneEstimate:
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 3:
        return ToneEstimate(tone="", confidence=0.0, contour_hz=[])

    third = max(1, len(voiced) // 3)
    start_hz = float(voiced[:third].mean())
    end_hz = float(voiced[-third:].mean())
    mean_hz = float(voiced.mean())

    f0_min = float(voiced.min())
    f0_max = float(voiced.max())
    f0_range = max(f0_max - f0_min, 1.0)
    delta = (end_hz - start_hz) / f0_range
    flat = f0_range < max(18.0, 0.10 * mean_hz)

    if flat:
        tone = "T1" if mean_hz > 170.0 else "T3"
        confidence = 0.55
    elif delta >= 0.15:
        tone = "T4"
        confidence = min(0.95, 0.6 + abs(delta))
    elif delta <= -0.15:
        tone = "T5"
        confidence = min(0.95, 0.6 + abs(delta))
    elif mean_hz > 170:
        tone = "T1"
        confidence = 0.7
    else:
        tone = "T2" if mean_hz > 130 else "T3"
        confidence = 0.65

    return ToneEstimate(
        tone=tone,
        confidence=round(float(confidence), 3),
        contour_hz=[float(v) for v in voiced],
    )


def estimate_segment_tone(audio_segment: np.ndarray, sr: int = 16000) -> ToneEstimate:
    if len(audio_segment) < int(0.08 * sr):
        return ToneEstimate(tone="", confidence=0.0, contour_hz=[])
    f0 = estimate_f0_contour(audio_segment, sr=sr)
    return classify_tone(f0)
