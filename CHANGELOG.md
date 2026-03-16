# CHANGELOG

## 2026-03-16

- Refactored token configuration to use `HF_TOKEN` and added `GH_TOKEN`.
- Added tone analysis module (`pipeline/tone.py`) for initial F0-based tone guesses.
- Updated `main.py` to:
  - apply initial tone estimates,
  - use `REVIEW_CONFIDENCE_THRESHOLD`,
  - generate `consonant&vowel&tones-inventory.csv`.
- Added `write_inventory_csv()` to `pipeline/output.py`.
- Replaced CLI-style review flow with browser-based review tooling in `review.py`:
  - generates `output/review_site/index.html`,
  - includes segment spectrogram + pitch contour,
  - generates vowel-zoom spectrogram/pitch views when phoneme timing is available,
  - saves decisions to `decisions.json`,
  - supports applying saved decisions back into `results.csv`.
- Added `publish_github.py` to initialize/publish repository `Rengmitca IPA` using `GH_TOKEN`.
- Updated `README.md` for new HTML review workflow, inventory output, and GitHub publishing instructions.
- Added `matplotlib` dependency and cleaned duplicate dependency entry in `requirements.txt`.
