"""
Central configuration for the Rengmitca IPA transcription pipeline.
Edit these settings before running main.py.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Device (GPU/CPU)
# ---------------------------------------------------------------------------
# Auto-detect CUDA availability, fall back to CPU if unavailable
try:
    import torch
    USE_GPU = torch.cuda.is_available()
    DEVICE = "cuda" if USE_GPU else "cpu"
except (ImportError, RuntimeError):
    USE_GPU = False
    DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
INPUT_DIR = Path("audio")      # Place .wav files here before running
OUTPUT_DIR = Path("output")    # TextGrids, CSVs, and flagged.csv written here

# ---------------------------------------------------------------------------
# ASR model  (Meta MMS — XLS-R architecture with Mizo language adapter)
# ---------------------------------------------------------------------------
MMS_MODEL_ID = "facebook/mms-1b-all"   # 1B-param model; has Mizo adapter
MMS_LANGUAGE = "miz"                   # Mizo/Lushai — closest available Tibeto-Burman ASR

# ---------------------------------------------------------------------------
# Speaker diarization (pyannote)
# ---------------------------------------------------------------------------
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"

# HuggingFace access token — required to download pyannote model.
# Get a free token at https://hf.co/settings/tokens and accept the model
# terms at https://hf.co/pyannote/speaker-diarization-3.1
# Set this string directly, or set the HF_TOKEN environment variable.
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

# GitHub token (used by publish_github.py)
GH_TOKEN: str | None = os.environ.get("GH_TOKEN")

# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16_000          # MMS expects 16 kHz mono
MIN_SEGMENT_DURATION = 0.3           # Skip diarized segments shorter than this (seconds)

# ---------------------------------------------------------------------------
# Confidence & review
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.6   # Segments with confidence < threshold → flagged.csv
REVIEW_CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Force alignment (phoneme and word level)
# ---------------------------------------------------------------------------
# Enable force alignment extraction from CTC emissions
# When enabled, extracts phoneme-level and word-level timestamps from the
# MMS model's CTC (Connectionist Temporal Classification) output
ENABLE_FORCE_ALIGNMENT = True

# Minimum phoneme duration (seconds) - filter out very short detections
# Typical CTC frame resolution is ~20ms, so this filters noise
PHONEME_MIN_DURATION = 0.02  # 20ms

# ---------------------------------------------------------------------------
# Rengmitca phoneme inventory (used by phoneme_map.py)
# ---------------------------------------------------------------------------
RENGMITCA_CONSONANTS = set(
    "p t c k ʔ pʰ tʰ ɟ m n ŋ ʃ s sʰ h l ɹ w j".split()
)
RENGMITCA_VOWELS = set(
    "i e æ a ɘ o ɑ ɔ u ɤ".split()
)
RENGMITCA_INVENTORY = RENGMITCA_CONSONANTS | RENGMITCA_VOWELS
# ---------------------------------------------------------------------------
# Rengmitca tones (for manual annotation in review.py)
# ---------------------------------------------------------------------------
# Tone inventory — currently placeholders pending linguistic documentation
# Update these with actual Rengmitca tones
RENGMITCA_TONES = {
    "T1": "high",
    "T2": "mid",
    "T3": "low",
    "T4": "rising",
    "T5": "falling",
    "": "unmarked",  # default: no tone specified
}

# Tone analysis defaults for automatic initial guess
TONE_F0_MIN_HZ = 65.0
TONE_F0_MAX_HZ = 450.0
