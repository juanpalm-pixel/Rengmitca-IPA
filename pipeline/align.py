"""
Force alignment extraction using CTC-based Wav2Vec2 emissions.

Extracts phoneme-level and word-level timestamps from the MMS model's
CTC (Connectionist Temporal Classification) output logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from transformers import AutoProcessor


@dataclass
class PhonemeToken:
    """Single phoneme with timing and confidence."""
    phone: str          # Mizo character or grapheme (before IPA mapping)
    start: float        # seconds
    end: float          # seconds
    confidence: float   # 0-1, from CTC emission probability


@dataclass
class AlignedWord:
    """Word with IPA transcription and timing."""
    word: str           # Original Mizo word
    ipa: str            # IPA transcription of word
    start: float        # seconds
    end: float          # seconds
    confidence: float   # mean confidence across phonemes in word


def extract_phoneme_alignment(
    logits: "torch.Tensor",
    predicted_ids: "torch.Tensor",
    processor: "AutoProcessor",
    seg_start: float,
    sample_rate: int = 16_000,
) -> list[PhonemeToken]:
    """
    Extract phoneme-level timestamps from CTC logits.

    This function performs CTC alignment by:
    1. Getting frame-level token predictions from logits
    2. Removing CTC blanks and collapsing consecutive duplicates
    3. Computing per-frame confidence scores
    4. Converting frame indices to time boundaries
    5. Decoding tokens to characters/graphemes

    Parameters
    ----------
    logits        : CTC logits tensor, shape (batch=1, frames, vocab_size)
    predicted_ids : Predicted token IDs, shape (batch=1, frames)
    processor     : Transformers processor for token decoding
    seg_start     : Segment start time in seconds (offset for timestamps)
    sample_rate   : Audio sample rate (default 16kHz)

    Returns
    -------
    List of PhonemeToken with character-level timing and confidence.
    """
    import torch
    import config

    # Get model config for frame timing calculations
    # inputs_to_logits_ratio tells us how many audio samples per logit frame
    # For Wav2Vec2/MMS, this is typically 320 (20ms frames at 16kHz)
    model_config = processor.feature_extractor
    if hasattr(model_config, 'inputs_to_logits_ratio'):
        inputs_to_logits_ratio = model_config.inputs_to_logits_ratio
    else:
        # Fallback: typical Wav2Vec2 ratio is 320 samples per frame
        inputs_to_logits_ratio = 320

    # Time per frame in seconds
    time_per_frame = inputs_to_logits_ratio / sample_rate

    # Get frame-level predictions and probabilities
    # logits shape: (1, frames, vocab_size)
    probs = torch.softmax(logits[0], dim=-1)  # (frames, vocab_size)
    frame_predictions = predicted_ids[0].cpu().numpy()  # (frames,)
    
    # Get pad token and blank token IDs
    pad_token_id = processor.tokenizer.pad_token_id
    # In CTC, blank is usually token 0, but let's be safe
    blank_token_id = 0
    
    # Extract confidence per frame (probability of predicted token)
    frame_confidences = []
    for i, token_id in enumerate(frame_predictions):
        if i < probs.shape[0]:
            conf = float(probs[i, token_id].item())
            frame_confidences.append(conf)
        else:
            frame_confidences.append(0.0)
    
    # CTC alignment: collapse repeated tokens and remove blanks
    # Result: list of (token_id, start_frame, end_frame, confidence)
    alignments = []
    prev_token = None
    start_frame = 0
    accumulated_conf = []
    
    for frame_idx, (token_id, conf) in enumerate(zip(frame_predictions, frame_confidences)):
        # Skip padding
        if token_id == pad_token_id:
            continue
            
        # Skip blank tokens (CTC blank)
        if token_id == blank_token_id:
            # If we were tracking a token, save it
            if prev_token is not None and prev_token != blank_token_id:
                mean_conf = np.mean(accumulated_conf) if accumulated_conf else 0.0
                alignments.append((prev_token, start_frame, frame_idx, mean_conf))
                accumulated_conf = []
            prev_token = blank_token_id
            continue
        
        # Token changed (and not to blank)
        if token_id != prev_token:
            # Save previous token if exists
            if prev_token is not None and prev_token != blank_token_id:
                mean_conf = np.mean(accumulated_conf) if accumulated_conf else 0.0
                alignments.append((prev_token, start_frame, frame_idx, mean_conf))
            # Start new token
            start_frame = frame_idx
            accumulated_conf = [conf]
            prev_token = token_id
        else:
            # Same token continues (CTC duplicate)
            accumulated_conf.append(conf)
    
    # Don't forget the last token
    if prev_token is not None and prev_token != blank_token_id and accumulated_conf:
        mean_conf = np.mean(accumulated_conf)
        alignments.append((prev_token, start_frame, len(frame_predictions), mean_conf))
    
    # Convert to PhonemeToken objects with actual characters
    phoneme_tokens = []
    for token_id, start_f, end_f, conf in alignments:
        # Decode token to character/grapheme
        try:
            char = processor.tokenizer.decode([token_id])
            # Clean up any extra spaces
            char = char.strip()
            if not char:
                continue
        except Exception:
            # If decode fails, skip this token
            continue
        
        # Convert frame indices to time
        start_time = seg_start + (start_f * time_per_frame)
        end_time = seg_start + (end_f * time_per_frame)
        
        # Filter out very short phonemes based on config
        duration = end_time - start_time
        if duration < config.PHONEME_MIN_DURATION:
            continue
        
        phoneme_tokens.append(PhonemeToken(
            phone=char,
            start=round(start_time, 4),
            end=round(end_time, 4),
            confidence=round(float(conf), 4),
        ))
    
    return phoneme_tokens


def map_to_ipa_phonemes(
    phoneme_tokens: list[PhonemeToken],
) -> list[PhonemeToken]:
    """
    Map Mizo character-level alignments to IPA phonemes.
    
    Handles multi-character graphemes (e.g., "ng" → "ŋ") by:
    - Merging consecutive character tokens that form a grapheme
    - Distributing time across the merged grapheme
    
    Parameters
    ----------
    phoneme_tokens : List of character-level PhonemeToken objects
    
    Returns
    -------
    List of PhonemeToken with IPA phonemes instead of Mizo characters.
    """
    from pipeline.phoneme_map import mizo_to_ipa
    
    # Reconstruct the text from phoneme tokens
    text = "".join(p.phone for p in phoneme_tokens)
    
    # Get IPA phones
    ipa_phones = mizo_to_ipa(text)
    
    # Map IPA phones back to timing
    # Strategy: distribute character-level timings across IPA phones
    # This is approximate since multi-char graphemes complicate the mapping
    
    if not ipa_phones:
        return []
    
    # Simple proportional distribution
    # Total duration divided by number of IPA phones
    if not phoneme_tokens:
        return []
    
    total_start = phoneme_tokens[0].start
    total_end = phoneme_tokens[-1].end
    total_duration = total_end - total_start
    
    # If we have the same number of IPA phones as character tokens, map 1:1
    if len(ipa_phones) == len(phoneme_tokens):
        return [
            PhonemeToken(
                phone=ipa_phone,
                start=char_token.start,
                end=char_token.end,
                confidence=char_token.confidence,
            )
            for ipa_phone, char_token in zip(ipa_phones, phoneme_tokens)
        ]
    
    # Otherwise, distribute time proportionally
    # This is a simplification - more sophisticated alignment could use DTW
    ipa_tokens = []
    time_per_phone = total_duration / len(ipa_phones) if ipa_phones else 0
    
    # Use average confidence across all character tokens
    avg_confidence = np.mean([p.confidence for p in phoneme_tokens])
    
    for i, ipa_phone in enumerate(ipa_phones):
        start = total_start + (i * time_per_phone)
        end = total_start + ((i + 1) * time_per_phone)
        
        ipa_tokens.append(PhonemeToken(
            phone=ipa_phone,
            start=round(start, 4),
            end=round(end, 4),
            confidence=round(float(avg_confidence), 4),
        ))
    
    return ipa_tokens


def process_word_alignment(
    word_tokens: list,  # List of WordToken from transcribe.py
) -> list[AlignedWord]:
    """
    Process word-level alignments and add IPA transcriptions.
    
    Parameters
    ----------
    word_tokens : List of WordToken objects with word, start, end
    
    Returns
    -------
    List of AlignedWord with IPA transcriptions and timing.
    """
    from pipeline.phoneme_map import mizo_to_ipa
    
    aligned_words = []
    for wt in word_tokens:
        # Convert word to IPA
        ipa_phones = mizo_to_ipa(wt.word)
        ipa_string = " ".join(ipa_phones)
        
        # Estimate confidence (will be refined later if we have phoneme-level data)
        # For now, use a placeholder
        confidence = 1.0
        
        aligned_words.append(AlignedWord(
            word=wt.word,
            ipa=ipa_string,
            start=wt.start,
            end=wt.end,
            confidence=confidence,
        ))
    
    return aligned_words
