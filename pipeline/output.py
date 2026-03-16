"""
Output writers for the transcription pipeline.

  write_csv        — appends rows to results.csv / flagged.csv
  write_textgrid   — writes a Praat TextGrid with Speaker + IPA tiers
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from praatio import textgrid as tg
from praatio.utilities.constants import Interval


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "filename",
    "speaker",
    "start",
    "end",
    "ipa_transcription",
    "ipa_with_flags",
    "mizo_text",
    "confidence",
    "tone",
]

# Phoneme-level CSV columns
PHONEME_CSV_COLUMNS = [
    "filename",
    "speaker",
    "start",
    "end",
    "phoneme",
    "confidence",
]

# Word-level CSV columns
WORD_CSV_COLUMNS = [
    "filename",
    "speaker",
    "start",
    "end",
    "word",
    "ipa_word",
    "confidence",
]


def append_to_csv(rows: list[dict], csv_path: Path) -> None:
    """
    Append *rows* (list of dicts matching CSV_COLUMNS) to *csv_path*.
    Creates the file with headers if it does not exist.
    """
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=write_header)


def write_phonemes_csv(phoneme_rows: list[dict], csv_path: Path) -> None:
    """
    Append phoneme-level alignment data to CSV.
    
    Parameters
    ----------
    phoneme_rows : list of dicts with keys matching PHONEME_CSV_COLUMNS
    csv_path     : path to phonemes.csv
    """
    if not phoneme_rows:
        return
    df = pd.DataFrame(phoneme_rows, columns=PHONEME_CSV_COLUMNS)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=write_header)


def write_words_csv(word_rows: list[dict], csv_path: Path) -> None:
    """
    Append word-level alignment data to CSV.
    
    Parameters
    ----------
    word_rows : list of dicts with keys matching WORD_CSV_COLUMNS
    csv_path  : path to words.csv
    """
    if not word_rows:
        return
    df = pd.DataFrame(word_rows, columns=WORD_CSV_COLUMNS)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=write_header)


def write_inventory_csv(results_csv: Path, inventory_path: Path) -> None:
    """
    Build a consonant/vowel/tone inventory with counts from results.csv.
    """
    if not results_csv.exists():
        return

    import config

    df = pd.read_csv(results_csv)
    counts: dict[tuple[str, str], int] = {}

    for _, row in df.iterrows():
        ipa = str(row.get("ipa_transcription", "")).strip()
        tone = str(row.get("tone", "")).strip()

        for phone in [p for p in ipa.split() if p]:
            if phone in config.RENGMITCA_CONSONANTS:
                key = ("consonant", phone)
            elif phone in config.RENGMITCA_VOWELS:
                key = ("vowel", phone)
            else:
                key = ("other", phone)
            counts[key] = counts.get(key, 0) + 1

        if tone:
            key = ("tone", tone)
            counts[key] = counts.get(key, 0) + 1

    rows = [
        {"category": category, "symbol": symbol, "count": count}
        for (category, symbol), count in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1]))
    ]
    pd.DataFrame(rows, columns=["category", "symbol", "count"]).to_csv(inventory_path, index=False)


# ---------------------------------------------------------------------------
# TextGrid
# ---------------------------------------------------------------------------

def write_textgrid(
    segments: list[dict],
    audio_duration: float,
    textgrid_path: Path,
    word_data: list[dict] | None = None,
    phoneme_data: list[dict] | None = None,
) -> None:
    """
    Write a Praat TextGrid with multiple tiers per speaker.
    
    For each speaker, creates:
    1. Segment tier - speaker segments with full IPA transcription
    2. Word tier - word-level alignments with IPA (if word_data provided)
    3. Phoneme tier - phoneme-level alignments (if phoneme_data provided)

    Parameters
    ----------
    segments       : list of dicts with keys:
                       speaker, start, end, ipa_transcription
    audio_duration : total duration of the source .wav (seconds)
    textgrid_path  : destination path for the .TextGrid file
    word_data      : optional list of dicts with keys:
                       speaker, start, end, word, ipa_word
    phoneme_data   : optional list of dicts with keys:
                       speaker, start, end, phoneme, confidence
    """
    from collections import defaultdict
    
    # Extract unique speakers and group data by speaker
    speaker_segments = defaultdict(list)
    speaker_words = defaultdict(list)
    speaker_phonemes = defaultdict(list)
    
    for seg in segments:
        speaker_segments[str(seg["speaker"])].append(seg)
    
    if word_data:
        for word in word_data:
            speaker_words[str(word["speaker"])].append(word)
    
    if phoneme_data:
        for phone in phoneme_data:
            speaker_phonemes[str(phone["speaker"])].append(phone)
    
    unique_speakers = sorted(speaker_segments.keys())
    
    grid = tg.Textgrid()
    
    # Create tiers for each speaker
    for speaker in unique_speakers:
        # 1. Segment tier (existing functionality)
        seg_intervals = []
        for seg in speaker_segments[speaker]:
            start = float(seg["start"])
            end = float(seg["end"])
            ipa = str(seg["ipa_transcription"]) if seg["ipa_transcription"] else ""
            label = f"{start:.4f} | {ipa}" if ipa else f"{start:.4f} |"
            seg_intervals.append(Interval(start, end, label))
        
        seg_tier = tg.IntervalTier(speaker, seg_intervals, minT=0.0, maxT=audio_duration)
        grid.addTier(seg_tier)
        
        # 2. Word tier (if word data provided)
        if speaker in speaker_words and speaker_words[speaker]:
            word_intervals = []
            for word in speaker_words[speaker]:
                start = float(word["start"])
                end = float(word["end"])
                word_text = str(word["word"])
                ipa_word = str(word.get("ipa_word", ""))
                # Label: "word | i p a"
                label = f"{word_text} | {ipa_word}" if ipa_word else word_text
                word_intervals.append(Interval(start, end, label))
            
            word_tier = tg.IntervalTier(
                f"{speaker}_words",
                word_intervals,
                minT=0.0,
                maxT=audio_duration
            )
            grid.addTier(word_tier)
        
        # 3. Phoneme tier (if phoneme data provided)
        if speaker in speaker_phonemes and speaker_phonemes[speaker]:
            phoneme_intervals = []
            for phone in speaker_phonemes[speaker]:
                start = float(phone["start"])
                end = float(phone["end"])
                phoneme = str(phone["phoneme"])
                conf = float(phone.get("confidence", 0.0))
                # Label: "p [0.92]"
                label = f"{phoneme} [{conf:.2f}]"
                phoneme_intervals.append(Interval(start, end, label))
            
            phoneme_tier = tg.IntervalTier(
                f"{speaker}_phonemes",
                phoneme_intervals,
                minT=0.0,
                maxT=audio_duration
            )
            grid.addTier(phoneme_tier)
    
    textgrid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(
        str(textgrid_path),
        format="short_textgrid",
        includeBlankSpaces=True,
    )
