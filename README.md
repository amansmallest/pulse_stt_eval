# Pulse STT vs Sarvam — Hindi/Hinglish eval

Evaluation script that runs **Smallest AI Pulse STT** (production streaming WebSocket) against a curated Hindi/Hinglish test pack, and computes WER side-by-side with the included **Sarvam saaras:v3** hypotheses on the same audios.

> **TL;DR** — Download the audio pack, set your Pulse API key, run one command, get a comparison table.

---

## 1. Download the data pack

The data pack contains ~13k audios + reference transcripts + Sarvam hypotheses for 22 datasets (15 robustness categories, 4 benchmarks, 3 Hinglish sets).

**Direct link:** https://drive.google.com/file/d/1xJMOyHnM-e10Ck6o9AfKd07iXJnDNnoa/view?usp=sharing
(6.3 GB tarball — extracts to ~6.7 GB)

### Option A — via `gdown` (recommended for CLI)
```bash
pip install -U gdown          # 6.x+; URL parsing is automatic
gdown "https://drive.google.com/file/d/1xJMOyHnM-e10Ck6o9AfKd07iXJnDNnoa/view?usp=sharing" \
      -O evals_test.tar.gz
```

### Option B — browser
Click the link → "Download anyway" (Drive will warn it's too big to virus-scan, that's fine) → save as `evals_test.tar.gz`.

### Extract
```bash
tar xzf evals_test.tar.gz
```

You'll get an `evals_test/` directory with:
```
evals_test/
├── README.md
├── summary.json
├── categories/   (15 datasets, ~4.9k audios)
├── benchmarks/   (4 datasets — fleurs, vistaar, mucs, gramvaani — ~7.4k audios)
└── hinglish/     (3 datasets — coshe500, hiacc-adult, hiacc-children — ~813 audios)
```

Each dataset directory contains:
- `manifest.jsonl` — `{audio_filepath, text, ...}` ground truth
- `our_hyps.jsonl` — Pulse STT hypothesis (precomputed)
- `sarvam_hyps.jsonl` — Sarvam saaras:v3 hypothesis (precomputed)
- `audio/` — the actual audio files (.wav/.flac), referenced by relative path in the manifest

---

## 2. Install dependencies

```bash
git clone https://github.com/<your-org>/pulse_stt_eval.git
cd pulse_stt_eval
pip install -r requirements.txt
```

---

## 3. Set your Pulse STT API key

Get a key from your Smallest AI dashboard, then:
```bash
export PULSE_API_KEY=sk_xxx
```

---

## 4. Run the eval

### Full pack (all 22 datasets, ~13k audios)
```bash
python run_pulse_eval.py --pack-dir ./evals_test
```

Takes ~90 minutes at default concurrency (8 parallel WebSocket sessions). Increase `--concurrency 16` if your network can handle it.

### Quick sanity check (one dataset)
```bash
python run_pulse_eval.py --pack-dir ./evals_test --only categories/repetition
```

### Other useful flags
```bash
# Only re-score (skip transcription, use existing pulse_hyps.jsonl)
python run_pulse_eval.py --pack-dir ./evals_test --skip-transcribe

# Use a different language code
python run_pulse_eval.py --pack-dir ./evals_test --language en
```

---

## 5. Output

For each dataset, the script writes a `pulse_hyps.jsonl` alongside the existing `our_hyps.jsonl` and `sarvam_hyps.jsonl`:
```
evals_test/categories/accent/
├── manifest.jsonl
├── our_hyps.jsonl         (precomputed Pulse hyps in the shipped pack)
├── sarvam_hyps.jsonl
├── pulse_hyps.jsonl       ← NEW: what your live run produced
└── audio/
```

It also prints a summary table:
```
Dataset                                  N    Pulse   Sarvam   Δ Pulse-Sar
─────────────────────────────────────────────────────────────────────────
  categories/accent                    199   17.01%   19.02%       -2.01
  categories/audio_quality              66    8.71%   17.58%       -8.87
  categories/boundary                  279    9.80%   18.80%       -9.00
  ...
```

…and saves the full table to `evals_test/pulse_summary.json`.

---

## How it works

- Streams 16 kHz mono PCM in 160 ms chunks to `wss://api.smallest.ai/waves/v1/pulse/get_text`
- Sends `{"type":"finalize"}` after the last chunk
- Collects all `is_final` segments and joins them as the hypothesis
- WER computed via `jiwer.process_words` after passing both refs and hyps through `normalize_indic()`:
  - nukta strip + chandrabindu→anusvara folding (via `indic-nlp-library`)
  - digit-run expansion to Indic words (via `indic-numtowords`)
  - Latin↔Devanagari code-switch equivalence (bundled lookup map)
  - punctuation strip + multi-word phrase rules

Same normalizer is used to score Sarvam's bundled hypotheses, so the comparison is apples-to-apples.

---

## Files in this repo

| File | Purpose |
|---|---|
| `run_pulse_eval.py` | Main entrypoint — streams audio to Pulse, scores against refs |
| `normalize.py` | `normalize_indic()` — shared text normalizer for Hindi/Indic WER |
| `_dict_blob.py` | Compiled Latin↔Devanagari equivalence data (internal, loaded by `normalize.py`) |
| `requirements.txt` | Python dependencies |

---

## License

MIT
