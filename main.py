"""
Rengmitca IPA Transcription Pipeline — batch entry point.

Usage:
    python main.py [--audio-dir AUDIO_DIR] [--output-dir OUTPUT_DIR]

Place .wav files in the audio directory (default: ./audio/) and run.
Results are written to the output directory (default: ./output/):
  - output/{stem}.TextGrid   per-file Praat TextGrid
  - output/results.csv       all segments
  - output/flagged.csv       low-confidence segments for manual review
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import soundfile as sf

# Suppress torchcodec warnings (we use soundfile for audio loading instead)
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import config
from pipeline.diarize import diarize
from pipeline.transcribe import load_audio, transcribe_segment
from pipeline.phoneme_map import text_to_rengmitca_ipa
from pipeline.output import append_to_csv, write_textgrid, write_phonemes_csv, write_words_csv, write_inventory_csv
from pipeline.tone import estimate_segment_tone


def resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    candidate_roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
        Path("C:/Program Files"),
        Path("C:/Program Files (x86)"),
    ]

    for root in candidate_roots:
        if not root.exists():
            continue
        matches = sorted(root.glob("**/ffmpeg.exe"))
        if matches:
            return str(matches[0])

    return None


def is_git_lfs_pointer(file_path: Path) -> bool:
    try:
        with file_path.open("rb") as handle:
            return handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1\n")
    except OSError:
        return False


def ensure_readable_wav(wav_path: Path) -> Path:
    if is_git_lfs_pointer(wav_path):
        raise RuntimeError(
            f"{wav_path.name} is a Git LFS pointer, not real audio. "
            "Run 'git lfs pull --include=\"audio/*\"' to download the actual recording."
        )

    try:
        sf.info(str(wav_path))
        return wav_path
    except Exception:
        ffmpeg = resolve_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                f"Unsupported WAV codec for {wav_path.name}. "
                "Install ffmpeg or convert the file to PCM16 WAV."
            )

        fixed_path = wav_path.with_name(f"{wav_path.stem}_pcm16.wav")
        proc = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(wav_path),
                "-ac",
                "1",
                "-ar",
                str(config.TARGET_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(fixed_path),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed for {wav_path.name}: {proc.stderr.strip()}")

        sf.info(str(fixed_path))
        print(f"  Converted unsupported WAV -> {fixed_path.name}")
        return fixed_path


def process_file(
    wav_path: Path,
    output_dir: Path,
    results_csv: Path,
    flagged_csv: Path,
    phonemes_csv: Path,
    words_csv: Path,
) -> None:
    print(f"\n-- {wav_path.name} --")

    original_wav_path = wav_path
    wav_path = ensure_readable_wav(wav_path)

    # Audio duration (needed for TextGrid maxT)
    info = sf.info(str(wav_path))
    duration = info.duration

    # Load full audio once (reused for all segments)
    audio = load_audio(wav_path, target_sr=config.TARGET_SAMPLE_RATE)

    # Diarization
    print("  Diarizing ...")
    try:
        segments = diarize(
            wav_path,
            model_id=config.DIARIZATION_MODEL_ID,
            hf_token=config.HF_TOKEN,
            min_duration=config.MIN_SEGMENT_DURATION,
        )
    except Exception as exc:
        print(f"  [!] Diarization failed: {exc}", file=sys.stderr)
        # Fall back to a single segment spanning the whole file
        from pipeline.diarize import DiarizedSegment
        segments = [DiarizedSegment(speaker="SPEAKER_00", start=0.0, end=duration)]

    print(f"  {len(segments)} speaker segment(s) found.")

    result_rows: list[dict] = []
    flagged_rows: list[dict] = []
    word_rows: list[dict] = []
    phoneme_rows: list[dict] = []

    for seg in segments:
        # ASR
        asr = transcribe_segment(
            audio,
            seg_start=seg.start,
            seg_end=seg.end,
            model_id=config.MMS_MODEL_ID,
            language=config.MMS_LANGUAGE,
            sample_rate=config.TARGET_SAMPLE_RATE,
        )

        # Phoneme mapping
        mapping = text_to_rengmitca_ipa(asr.text)
        seg_audio = audio[int(seg.start * config.TARGET_SAMPLE_RATE): int(seg.end * config.TARGET_SAMPLE_RATE)]
        tone_estimate = estimate_segment_tone(seg_audio, sr=config.TARGET_SAMPLE_RATE)

        # Combined confidence: ASR confidence × (1 − flagged fraction)
        combined_conf = round(asr.confidence * (1.0 - mapping.flagged_fraction), 4)

        row = {
            "filename":         original_wav_path.name,
            "speaker":          seg.speaker,
            "start":            seg.start,
            "end":              seg.end,
            "ipa_transcription": mapping.ipa_string,
            "ipa_with_flags":   mapping.ipa_string_with_flags,
            "mizo_text":        asr.text,
            "confidence":       combined_conf,
            "tone":             tone_estimate.tone or mapping.tone,
        }

        result_rows.append(row)
        if combined_conf < config.REVIEW_CONFIDENCE_THRESHOLD or not asr.text:
            flagged_rows.append(row)

        # Collect word-level alignment data
        from pipeline.align import process_word_alignment
        if asr.words:
            aligned_words = process_word_alignment(asr.words)
            for aw in aligned_words:
                word_rows.append({
                    "filename": original_wav_path.name,
                    "speaker": seg.speaker,
                    "start": aw.start,
                    "end": aw.end,
                    "word": aw.word,
                    "ipa_word": aw.ipa,
                    "confidence": aw.confidence,
                })

        # Collect phoneme-level alignment data (aligned to word boundaries)
        # Only include phonemes that fall within word time boundaries
        if asr.phonemes and asr.words:
            for word in asr.words:
                # Filter phonemes that fall within this word's time boundaries
                # Allow small tolerance for boundary matching (±10ms)
                word_phonemes = [
                    p for p in asr.phonemes
                    if (p.start >= word.start - 0.01 and p.end <= word.end + 0.01)
                ]
                for phone_token in word_phonemes:
                    phoneme_rows.append({
                        "filename": original_wav_path.name,
                        "speaker": seg.speaker,
                        "start": phone_token.start,
                        "end": phone_token.end,
                        "phoneme": phone_token.phone,
                        "confidence": phone_token.confidence,
                    })

        print(
            f"  [{seg.speaker}] {seg.start:.2f}–{seg.end:.2f}s | "
            f"conf={combined_conf:.2f} | {mapping.ipa_string or '(empty)'}"
        )

    # Write TextGrid with word and phoneme tiers
    tg_path = output_dir / f"{original_wav_path.stem}.TextGrid"
    write_textgrid(
        result_rows,
        duration,
        tg_path,
        word_data=word_rows if word_rows else None,
        phoneme_data=phoneme_rows if phoneme_rows else None,
    )
    print(f"  TextGrid → {tg_path}")

    # Append to CSVs
    if result_rows:
        append_to_csv(result_rows, results_csv)
    if flagged_rows:
        append_to_csv(flagged_rows, flagged_csv)
        print(f"  {len(flagged_rows)} segment(s) flagged for review.")
    
    # Write word and phoneme CSVs
    if word_rows:
        write_words_csv(word_rows, words_csv)
        print(f"  {len(word_rows)} word(s) aligned.")
    if phoneme_rows:
        write_phonemes_csv(phoneme_rows, phonemes_csv)
        print(f"  {len(phoneme_rows)} phoneme(s) aligned.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rengmitca IPA transcription pipeline")
    parser.add_argument("--audio-dir",  default=str(config.INPUT_DIR),  help="Directory of .wav files")
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    audio_dir  = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for required dependencies
    if not TORCH_AVAILABLE:
        print("\n[ERROR] PyTorch is not installed.")
        print("   This pipeline requires PyTorch to function.")
        print("\n   To install PyTorch and all dependencies, run:")
        print("   pip install -r requirements.txt")
        print("\n   Or for GPU support (NVIDIA CUDA):")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)

    # Detect GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[GPU] Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("[WARN] No GPU detected. Running on CPU.")

    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in '{audio_dir}'. Place your recordings there and re-run.")
        sys.exit(0)

    results_csv = output_dir / "results.csv"
    flagged_csv = output_dir / "flagged.csv"
    phonemes_csv = output_dir / "phonemes.csv"
    words_csv = output_dir / "words.csv"

    # Clear existing CSVs so headers are written fresh
    results_csv.unlink(missing_ok=True)
    flagged_csv.unlink(missing_ok=True)
    phonemes_csv.unlink(missing_ok=True)
    words_csv.unlink(missing_ok=True)

    print(f"Found {len(wav_files)} file(s). Starting pipeline ...")
    print(f"  ASR model : {config.MMS_MODEL_ID} (lang={config.MMS_LANGUAGE})")
    print(f"  Diarization: {config.DIARIZATION_MODEL_ID}")
    print(f"  Device: {device.upper()}")
    if not config.HF_TOKEN:
        print(
            "\n  [WARN] HF_TOKEN is not set. Diarization will fail unless you set it in\n"
            "     config.py or via the HF_TOKEN environment variable.\n"
            "     Get a free token at https://hf.co/settings/tokens\n"
        )
    else:
        print(f"  [OK] HF_TOKEN found (hf_...{config.HF_TOKEN[-8:]})")

    for wav_path in wav_files:
        process_file(wav_path, output_dir, results_csv, flagged_csv, phonemes_csv, words_csv)

    print(f"\nDone.  Results: {results_csv}")
    print(f"       Words:   {words_csv}")
    print(f"       Phonemes: {phonemes_csv}")
    inventory_path = output_dir / "consonant&vowel&tones-inventory.csv"
    write_inventory_csv(results_csv, inventory_path)
    print(f"       Inventory: {inventory_path}")
    if flagged_csv.exists():
        print(f"       Flagged: {flagged_csv}  ← run review.py --serve to review in browser")


if __name__ == "__main__":
    main()
