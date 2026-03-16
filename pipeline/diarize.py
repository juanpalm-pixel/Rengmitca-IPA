"""
Speaker diarization using pyannote/speaker-diarization-3.1.

Returns a list of (speaker_label, start_sec, end_sec) segments for a .wav file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DiarizedSegment:
    speaker: str
    start: float   # seconds
    end: float     # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


_pipeline = None  # cached diarization pipeline


def _load_pipeline(model_id: str, hf_token: str | None):
    """Lazy-load the pyannote diarization pipeline (heavy, load once)."""
    global _pipeline
    if _pipeline is None:
        from pyannote.audio import Pipeline
        import config
        print(f"[diarize] Loading {model_id} …")
        print(f"[diarize] Using device: {config.DEVICE}")
        _pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
        _pipeline = _pipeline.to(config.DEVICE)  # Move pipeline to GPU or CPU
        print("[diarize] Pipeline ready.")
    return _pipeline


def diarize(
    wav_path: Path,
    model_id: str,
    hf_token: str | None,
    min_duration: float = 0.3,
) -> list[DiarizedSegment]:
    """
    Run speaker diarization on *wav_path*.

    Parameters
    ----------
    wav_path     : path to the .wav file
    model_id     : pyannote pipeline model ID
    hf_token     : HuggingFace access token (required for pyannote models)
    min_duration : discard segments shorter than this many seconds

    Returns
    -------
    List of DiarizedSegment sorted by start time.
    """
    pipe = _load_pipeline(model_id, hf_token)
    diarization = pipe(str(wav_path))

    segments: list[DiarizedSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        if dur >= min_duration:
            segments.append(DiarizedSegment(
                speaker=speaker,
                start=round(turn.start, 4),
                end=round(turn.end, 4),
            ))

    segments.sort(key=lambda s: s.start)
    return segments
