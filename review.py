from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

import config


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _plot_spectrogram_with_pitch(audio: np.ndarray, sr: int, png_path: Path) -> None:
    f0, _, _ = librosa.pyin(
        audio,
        fmin=config.TONE_F0_MIN_HZ,
        fmax=config.TONE_F0_MAX_HZ,
        sr=sr,
        frame_length=2048,
        hop_length=256,
    )
    s = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    db = librosa.amplitude_to_db(s, ref=np.max)
    t_spec = np.arange(db.shape[1]) * (256 / sr)
    t_f0 = np.arange(len(f0)) * (256 / sr)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.imshow(
        db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, t_spec[-1] if len(t_spec) else 0, 0, sr / 2],
    )
    valid = ~np.isnan(f0)
    if np.any(valid):
        ax.plot(t_f0[valid], f0[valid], color="cyan", linewidth=1.5, label="Pitch (Hz)")
        ax.legend(loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram + Pitch Contour")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def _slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    s = max(0, int(start * sr))
    e = min(len(audio), int(end * sr))
    if e <= s:
        return np.array([], dtype=np.float32)
    return audio[s:e]


def build_review_site(
    flagged_csv: Path,
    results_csv: Path,
    audio_dir: Path,
    output_dir: Path,
    phonemes_csv: Path,
    site_dir: Path,
) -> Path:
    if flagged_csv.exists():
        review_df = pd.read_csv(flagged_csv)
    elif results_csv.exists():
        review_df = pd.read_csv(results_csv)
    else:
        raise FileNotFoundError("Neither flagged.csv nor results.csv exists.")

    phon_df = pd.read_csv(phonemes_csv) if phonemes_csv.exists() else pd.DataFrame()
    site_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = site_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    segments: list[dict] = []
    for idx, row in review_df.iterrows():
        filename = str(row["filename"])
        speaker = str(row["speaker"])
        start = float(row["start"])
        end = float(row["end"])
        conf = float(row.get("confidence", 0.0))
        wav_path = audio_dir / filename
        if not wav_path.exists():
            continue

        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio.mean(axis=1)
        segment_audio = _slice_audio(audio, sr, start, end)
        if len(segment_audio) == 0:
            continue

        base = _safe_name(f"{Path(filename).stem}_{speaker}_{idx}_{start:.3f}_{end:.3f}")
        clip_rel = Path("assets") / f"{base}.wav"
        full_spec_rel = Path("assets") / f"{base}_spec.png"
        sf.write(str(site_dir / clip_rel), segment_audio, sr)
        _plot_spectrogram_with_pitch(segment_audio, sr, site_dir / full_spec_rel)

        vowel_images: list[str] = []
        if not phon_df.empty:
            p = phon_df[
                (phon_df["filename"].astype(str) == filename)
                & (phon_df["speaker"].astype(str) == speaker)
                & (phon_df["start"].astype(float) >= start)
                & (phon_df["end"].astype(float) <= end)
                & (phon_df["phoneme"].astype(str).isin(config.RENGMITCA_VOWELS))
            ]
            for j, prow in p.iterrows():
                v_start = float(prow["start"])
                v_end = float(prow["end"])
                v_audio = _slice_audio(audio, sr, v_start, v_end)
                if len(v_audio) < int(0.03 * sr):
                    continue
                v_rel = Path("assets") / f"{base}_vowel_{j}.png"
                _plot_spectrogram_with_pitch(v_audio, sr, site_dir / v_rel)
                vowel_images.append(v_rel.as_posix())

        segments.append(
            {
                "id": base,
                "filename": filename,
                "speaker": speaker,
                "start": start,
                "end": end,
                "confidence": conf,
                "mizo_text": str(row.get("mizo_text", "")),
                "ipa_transcription": str(row.get("ipa_transcription", "")),
                "tone": str(row.get("tone", "")),
                "audio": clip_rel.as_posix(),
                "spectrogram": full_spec_rel.as_posix(),
                "vowel_images": vowel_images,
            }
        )

    (site_dir / "segments.json").write_text(json.dumps({"segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
    (site_dir / "decisions.json").write_text("[]", encoding="utf-8")
    (site_dir / "index.html").write_text(_review_html(), encoding="utf-8")
    return site_dir / "index.html"


def apply_decisions(results_csv: Path, decisions_json: Path) -> None:
    if not results_csv.exists() or not decisions_json.exists():
        return

    df = pd.read_csv(results_csv)
    decisions = json.loads(decisions_json.read_text(encoding="utf-8"))
    if not isinstance(decisions, list):
        return

    key_cols = ["filename", "speaker", "start", "end"]
    indexed = df.set_index(key_cols)
    for d in decisions:
        key = (str(d["filename"]), str(d["speaker"]), float(d["start"]), float(d["end"]))
        if key in indexed.index:
            indexed.at[key, "ipa_transcription"] = str(d.get("ipa_transcription", ""))
            indexed.at[key, "ipa_with_flags"] = str(d.get("ipa_transcription", ""))
            indexed.at[key, "tone"] = str(d.get("tone", ""))
    indexed.reset_index().to_csv(results_csv, index=False)


class _ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def do_POST(self):
        if self.path != "/api/save":
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return

        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(n).decode("utf-8")
            data = json.loads(payload)
            decisions_path = Path(self.directory) / "decisions.json"
            decisions_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
        except Exception:
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.end_headers()


def serve_review(site_dir: Path, port: int) -> None:
    handler = lambda *args, **kwargs: _ReviewHandler(*args, directory=str(site_dir), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Review UI: http://127.0.0.1:{port}/index.html")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def _review_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Rengmitca Review</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; max-width: 1100px; }
    .card { border: 1px solid #ccc; border-radius: 8px; padding: 14px; margin-bottom: 16px; }
    img { max-width: 100%; border-radius: 6px; border: 1px solid #ddd; margin-top: 8px; }
    input { width: 100%; padding: 6px; margin: 4px 0 8px 0; }
    button { padding: 8px 12px; margin-right: 8px; }
  </style>
</head>
<body>
  <h1>Rengmitca HTML Review</h1>
  <p>Inspect low-confidence segments, especially vowels with spectrogram + pitch contour, then save decisions.</p>
  <div>
    <button onclick="save()">Save decisions.json</button>
    <span id="status"></span>
  </div>
  <div id="root"></div>
  <script>
    const decisions = [];
    fetch('segments.json').then(r => r.json()).then(data => {
      const root = document.getElementById('root');
      data.segments.forEach(seg => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <h3>${seg.filename} | ${seg.speaker} | ${seg.start.toFixed(3)}-${seg.end.toFixed(3)}s</h3>
          <div>Confidence: ${seg.confidence.toFixed(3)} | Mizo: ${seg.mizo_text || '(empty)'}</div>
          <audio controls src="${seg.audio}"></audio>
          <div><img src="${seg.spectrogram}" alt="segment spectrogram" /></div>
          <div><strong>Vowel zooms</strong></div>
          <div>${seg.vowel_images.map(v => `<img src="${v}" alt="vowel zoom"/>`).join('')}</div>
          <label>IPA transcription</label>
          <input id="ipa_${seg.id}" value="${seg.ipa_transcription || ''}" />
          <label>Tone</label>
          <input id="tone_${seg.id}" value="${seg.tone || ''}" />
          <button onclick='capture(${JSON.stringify(seg)})'>Capture decision</button>
        `;
        root.appendChild(card);
      });
    });

    function capture(seg) {
      const ipa = document.getElementById(`ipa_${seg.id}`).value;
      const tone = document.getElementById(`tone_${seg.id}`).value;
      const idx = decisions.findIndex(d => d.id === seg.id);
      const item = { id: seg.id, filename: seg.filename, speaker: seg.speaker, start: seg.start, end: seg.end, ipa_transcription: ipa, tone };
      if (idx >= 0) decisions[idx] = item; else decisions.push(item);
      document.getElementById('status').textContent = `${decisions.length} decisions captured`;
    }

    function save() {
      fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(decisions)
      }).then(() => {
        document.getElementById('status').textContent = 'Saved to decisions.json';
      }).catch(() => {
        document.getElementById('status').textContent = 'Save failed';
      });
    }
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/serve browser-based review UI for Rengmitca outputs")
    parser.add_argument("--flagged", default="output/flagged.csv")
    parser.add_argument("--results", default="output/results.csv")
    parser.add_argument("--audio-dir", default="audio")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--phonemes", default="output/phonemes.csv")
    parser.add_argument("--site-dir", default="output/review_site")
    parser.add_argument("--serve", action="store_true", help="Serve the review UI locally")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--apply-decisions", action="store_true", help="Apply site decisions.json to results.csv")
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    build_review_site(
        flagged_csv=Path(args.flagged),
        results_csv=Path(args.results),
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output_dir),
        phonemes_csv=Path(args.phonemes),
        site_dir=site_dir,
    )
    print(f"Review site generated at {site_dir}\\index.html")

    if args.apply_decisions:
        apply_decisions(Path(args.results), site_dir / "decisions.json")
        print(f"Applied decisions from {site_dir}\\decisions.json to {args.results}")

    if args.serve:
        serve_review(site_dir=site_dir, port=args.port)


if __name__ == "__main__":
    main()
