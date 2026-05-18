"""
Speaker diarization using pyannote/speaker-diarization-3.1.

Returns a list of (speaker_label, start_sec, end_sec) segments for a .wav file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


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
        import os
        import torch
        from pyannote.audio import Pipeline
        import config

        device = torch.device(config.DEVICE)
        print(f"[diarize] Loading {model_id} …")
        print(f"[diarize] Using device: {device}")
        if hf_token:
            # Ensure downstream pyannote/speechbrain downloads see the token.
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            try:
                from huggingface_hub import login

                login(token=hf_token, add_to_git_credential=False, skip_if_logged_in=True)
            except Exception:
                pass

        # PyTorch 2.6+ defaults torch.load(weights_only=True). pyannote/lightning
        # checkpoints include richer objects and require classic loading behavior.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

        # PyTorch 2.6+ defaults torch.load(weights_only=True), which can reject
        # older checkpoint metadata classes used by pyannote/lightning artifacts.
        try:
            from torch.serialization import add_safe_globals
            from torch.torch_version import TorchVersion

            add_safe_globals([TorchVersion])
        except Exception:
            pass
        # pyannote/huggingface auth kwarg changed from use_auth_token -> token.
        try:
            _pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
        except TypeError:
            try:
                _pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load diarization pipeline: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Failed to load diarization pipeline: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        _pipeline = _pipeline.to(device)
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
    import torch

    pipe = _load_pipeline(model_id, hf_token)
    waveform, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    diarization = pipe(
        {
            "waveform": torch.from_numpy(np.ascontiguousarray(waveform.T)),
            "sample_rate": sample_rate,
        }
    )

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
