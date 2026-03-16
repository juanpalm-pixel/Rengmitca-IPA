# Rengmitca IPA Transcription Pipeline

A Python pipeline that produces IPA transcriptions of Rengmitca audio recordings using a pre-trained ASR model for the closely related Tibeto-Burman language Mizo (Lushai). Outputs are speaker-diarized Praat TextGrids and a CSV file.

---

## How it works

Rengmitca has no ASR system of its own. This pipeline approximates its phonology by:

1. **Separating speakers** in each recording using a neural diarization model
2. **Transcribing each speaker's speech** using Meta's MMS model trained on Mizo — the closest available Tibeto-Burman language
3. **Converting the Mizo output to IPA** via rule-based grapheme-to-phoneme (G2P) conversion
4. **Mapping the IPA phones to the Rengmitca inventory** using exact matches and linguistically motivated near-matches (e.g. Mizo /b/ → Rengmitca /p/, Mizo /r/ → /ɹ/, Mizo /tʃ/ → /c/)
5. **Flagging uncertain segments** (low confidence or unresolvable phones) for manual review

```
.wav file
   │
   ▼
[diarize]    pyannote/speaker-diarization-3.1
             → (SPEAKER_00, 0.0s, 2.3s), (SPEAKER_01, 2.5s, 5.1s), …
   │
   ▼
[transcribe] Meta MMS — facebook/mms-1b-all (Mizo adapter)
             → "lalpawl chite" + word timestamps + CTC confidence
   │
   ▼
[G2P]        Mizo Latin orthography → IPA
             → [l, a, l, p, a, w, l,  tʃ, i, t, e]
   │
   ▼
[map]        IPA → Rengmitca inventory
             → [l, a, l, p, a, w, l,  c,  i, t, e]
   │
   ▼
[output]     results.csv  +  {filename}.TextGrid
```

---

## Rengmitca phoneme inventory

| Class | Phones |
|-------|--------|
| Consonants | p  t  c  k  ʔ  pʰ  tʰ  ɟ  m  n  ŋ  ʃ  s  sʰ  h  l  ɹ  w  j |
| Vowels | i  e  æ  a  ɘ  o  ɑ  ɔ  u  ɣ |

Mizo phones with no exact match are substituted using the nearest Rengmitca phone. Phones with no reasonable substitute are marked `?` and the segment is flagged for review.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a HuggingFace token

The speaker diarization model requires a free HuggingFace account:

1. Create a token at <https://hf.co/settings/tokens>
2. Accept the model terms at <https://hf.co/pyannote/speaker-diarization-3.1>
3. Add your token to `config.py`:

```python
HF_TOKEN = "hf_your_token_here"
```

Or set it as an environment variable:

```bash
set HF_TOKEN=hf_your_token_here   # Windows
export HF_TOKEN=hf_your_token_here  # macOS/Linux
```

> **Note:** The MMS ASR model (~4 GB) and diarization model (~1 GB) are downloaded automatically on first run.

### 3. (Optional) Enable GPU acceleration

The pipeline detects CUDA availability and automatically uses GPU if available, otherwise falls back to CPU.

To enable GPU:
- **NVIDIA GPUs with CUDA:** Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (adjust `cu118` for your CUDA version)
- **Other GPUs:** Consult PyTorch documentation for your hardware

If GPU is unavailable, the pipeline runs on CPU (slower but still functional).

---

## Usage

### Step 1 — Add your recordings

Place all `.wav` files in the `audio/` folder.

### Step 2 — Run the pipeline

**Easy method (Windows):**
```bash
run.bat
```
This automatically uses the correct Python environment.

**Manual method:**
```bash
python main.py
```

Or if using the conda environment directly:
```bash
C:\Users\pablo\miniconda3\envs\ipa-transcriber-3\python.exe main.py
```

Optional arguments:

```
--audio-dir   PATH   Directory of .wav files  (default: audio/)
--output-dir  PATH   Where to write results   (default: output/)
```

### Step 3 — Review flagged segments in HTML

```bash
python review.py --serve
```

The browser review UI includes:
- segment audio playback
- full-segment spectrogram + pitch contour
- vowel-focused zoom spectrograms with pitch contour
- editable IPA and tone fields per segment
- save button writing `output/review_site/decisions.json`

Apply saved review decisions back into `results.csv`:

```bash
python review.py --apply-decisions
```

---

## Output files

All outputs are written to `output/`.

| File | Description |
|------|-------------|
| `results.csv` | Every segment: filename, speaker, start, end, IPA transcription, confidence, tone |
| `flagged.csv` | Subset of `results.csv` where confidence < 0.6, for manual review |
| `consonant&vowel&tones-inventory.csv` | Inventory table of observed consonants, vowels, and tones with counts |
| `review_site/index.html` | HTML review page for low-confidence segments |
| `{name}.TextGrid` | Praat TextGrid per recording with one tier per speaker (e.g., **SPEAKER_00**, **SPEAKER_01**). Each tier labels show the start timestamp and IPA transcription in the format `[start_time] \| IPA_transcription` |

### CSV columns

| Column | Description |
|--------|-------------|
| `filename` | Source `.wav` filename |
| `speaker` | Speaker label from diarization (e.g. `SPEAKER_00`) |
| `start` / `end` | Segment timestamps in seconds |
| `ipa_transcription` | Rengmitca IPA (uncertain phones removed) |
| `ipa_with_flags` | IPA with `?` markers for unresolvable phones |
| `mizo_text` | Raw Mizo ASR output before phoneme mapping |
| `confidence` | Score 0–1 (ASR confidence × phone-match rate) |
| `tone` | Tone annotation (manually added during review; initially empty) |

---

## Project structure

```
ipa-transcriber/
├── audio/                   ← place .wav files here
├── output/                  ← TextGrids and CSVs written here
├── pipeline/
│   ├── diarize.py           speaker diarization (pyannote)
│   ├── transcribe.py        MMS ASR with Mizo language adapter
│   ├── tone.py              F0 contour extraction + initial tone guess
│   ├── phoneme_map.py       Mizo G2P + Rengmitca inventory mapping
│   └── output.py            CSV and TextGrid writers
├── config.py                all configurable settings
├── main.py                  batch pipeline entry point
├── review.py                HTML review builder/server
├── publish_github.py        creates/pushes GitHub repo using GH_TOKEN
├── CHANGELOG.md
└── requirements.txt
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `DEVICE` | `"cuda"` or `"cpu"` | Device for model inference. Auto-detected based on CUDA availability (GPU if available, else CPU) |
| `USE_GPU` | `True` or `False` | Whether GPU acceleration is currently enabled |
| `INPUT_DIR` | `audio/` | Directory of input `.wav` files |
| `OUTPUT_DIR` | `output/` | Directory for results |
| `MMS_MODEL_ID` | `facebook/mms-1b-all` | HuggingFace model ID for ASR |
| `MMS_LANGUAGE` | `miz` | MMS language code (Mizo/Lushai) |
| `DIARIZATION_MODEL_ID` | `pyannote/speaker-diarization-3.1` | Diarization model |
| `HF_TOKEN` | `None` | HuggingFace access token |
| `GH_TOKEN` | `None` | GitHub token used by `publish_github.py` |
| `CONFIDENCE_THRESHOLD` | `0.6` | Segments below this go to `flagged.csv` |
| `MIN_SEGMENT_DURATION` | `0.3` | Discard diarized segments shorter than this (seconds) |
| `RENGMITCA_TONES` | Placeholders (T1–T5) | Tone inventory for manual annotation during review |

---

## Publish to GitHub (public repo `Rengmitca IPA`)

Set your token:

```bash
set GH_TOKEN=ghp_your_token_here
```

Run:

```bash
python publish_github.py --repo "Rengmitca IPA" --visibility public
```

---

## Limitations

- **No Rengmitca-specific model exists.** The Mizo ASR model is an approximation — accuracy will be imperfect, especially for phones unique to Rengmitca (æ, ɘ, ɣ, sʰ). These will typically be flagged for review.
- **Tone guesses are bootstrap-only.** Initial tone labels are estimated from pitch contour and should be verified in the HTML review interface.
- **First run is slow** due to model downloads (~5 GB of models). GPU acceleration (if available) significantly speeds up inference.
