"""
ASR transcription using Meta MMS (XLS-R architecture) with the Mizo language adapter.

Produces Mizo Latin-script text plus word-level timestamps and a confidence score
for each audio segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WordToken:
    word: str
    start: float   # seconds, relative to the start of the full audio file
    end: float     # seconds


@dataclass
class TranscriptionResult:
    text: str                          # raw Mizo text
    words: list[WordToken] = field(default_factory=list)
    phonemes: list = field(default_factory=list)  # list[PhonemeToken] from align.py
    confidence: float = 0.0            # mean CTC probability (0–1)


_processor = None
_model = None
_loaded_lang: str | None = None


def _load_model(model_id: str, language: str):
    """Lazy-load and cache the MMS processor + model with the requested language adapter."""
    global _processor, _model, _loaded_lang
    if _model is None or _loaded_lang != language:
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        import config
        print(f"[transcribe] Loading {model_id} with language adapter '{language}' …")
        print(f"[transcribe] Using device: {config.DEVICE}")
        _processor = AutoProcessor.from_pretrained(model_id)
        _model = Wav2Vec2ForCTC.from_pretrained(model_id)
        _model = _model.to(config.DEVICE)  # Move model to GPU or CPU
        _processor.tokenizer.set_target_lang(language)
        _model.load_adapter(language)
        _model.eval()
        _loaded_lang = language
        print("[transcribe] Model ready.")
    return _processor, _model


def _load_audio(wav_path: Path, target_sr: int = 16_000) -> np.ndarray:
    """Load a .wav file as a 16 kHz mono float32 array."""
    import soundfile as sf
    import librosa

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)   # stereo → mono
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def transcribe_segment(
    audio_full: np.ndarray,
    seg_start: float,
    seg_end: float,
    model_id: str,
    language: str,
    sample_rate: int = 16_000,
) -> TranscriptionResult:
    """
    Transcribe a time slice of *audio_full*.

    Parameters
    ----------
    audio_full  : full mono 16 kHz audio array
    seg_start   : segment start (seconds)
    seg_end     : segment end (seconds)
    model_id    : HuggingFace model ID
    language    : MMS language code (e.g. 'miz' for Mizo)
    sample_rate : expected to be 16_000

    Returns
    -------
    TranscriptionResult with Mizo text, word tokens, and confidence.
    """
    import torch
    import config
    processor, model = _load_model(model_id, language)

    start_frame = int(seg_start * sample_rate)
    end_frame = int(seg_end * sample_rate)
    segment = audio_full[start_frame:end_frame]

    if len(segment) < int(0.1 * sample_rate):
        return TranscriptionResult(text="", words=[], phonemes=[], confidence=0.0)

    inputs = processor(segment, sampling_rate=sample_rate, return_tensors="pt")
    # Move input tensors to the same device as the model
    inputs = {key: val.to(config.DEVICE) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    # Predicted token IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Mean of the max softmax probability per frame → confidence proxy
    probs = torch.softmax(logits, dim=-1)
    confidence = float(probs.max(dim=-1).values.mean().item())

    # Decode text
    transcription = processor.decode(predicted_ids[0])

    # Extract phoneme-level alignments from CTC emissions
    try:
        from pipeline.align import extract_phoneme_alignment, map_to_ipa_phonemes
        from pipeline.phoneme_map import map_aligned_phonemes_to_rengmitca
        
        phoneme_tokens = extract_phoneme_alignment(
            logits=logits,
            predicted_ids=predicted_ids,
            processor=processor,
            seg_start=seg_start,
            sample_rate=sample_rate,
        )
        # Map Mizo characters to IPA phonemes
        ipa_phonemes = map_to_ipa_phonemes(phoneme_tokens)
        # Map IPA phonemes to Rengmitca inventory
        rengmitca_phonemes = map_aligned_phonemes_to_rengmitca(ipa_phonemes)
    except Exception as e:
        # If alignment extraction fails, continue without phoneme data
        print(f"[transcribe] Warning: phoneme alignment failed: {e}")
        rengmitca_phonemes = []

    # Word-level timestamps (frame offsets → seconds)
    try:
        decoded = processor.tokenizer.decode(
            predicted_ids[0].tolist(),
            output_word_offsets=True,
        )
        # inputs_to_logits_ratio: samples per logit frame
        time_ratio = model.config.inputs_to_logits_ratio / sample_rate
        words = [
            WordToken(
                word=wo["word"],
                start=round(seg_start + wo["start_offset"] * time_ratio, 4),
                end=round(seg_start + wo["end_offset"] * time_ratio, 4),
            )
            for wo in decoded.word_offsets
        ]
    except Exception:
        # If word offsets unavailable, span the whole segment
        words = [WordToken(word=transcription, start=seg_start, end=seg_end)]

    return TranscriptionResult(
        text=transcription.strip(),
        words=words,
        phonemes=rengmitca_phonemes,
        confidence=confidence,
    )


def load_audio(wav_path: Path, target_sr: int = 16_000) -> np.ndarray:
    """Public helper to load a .wav into a 16 kHz mono float32 array."""
    return _load_audio(wav_path, target_sr)
