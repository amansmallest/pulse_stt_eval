"""Run Pulse STT (Smallest AI) over every dataset in the evals_acefone_test pack.

For each dataset under {pack-dir}/categories/, /benchmarks/, /hinglish/:
  - read manifest.jsonl
  - stream each audio to Pulse STT prod and collect the hypothesis
  - write pulse_hyps.jsonl alongside the existing our_hyps.jsonl / sarvam_hyps.jsonl
  - compute WER (Pulse vs Sarvam vs Reference) on the normalized text and print
    a side-by-side summary

Usage:
    export PULSE_API_KEY=sk_xxx
    python run_pulse_eval.py --pack-dir ./evals_acefone_test
    python run_pulse_eval.py --pack-dir ./evals_acefone_test --concurrency 8
    python run_pulse_eval.py --pack-dir ./evals_acefone_test --only categories/accent

Outputs:
    {dataset}/pulse_hyps.jsonl   — {audio_filepath, hypothesis} per line, line-aligned with manifest.jsonl
    {pack-dir}/pulse_summary.json — aggregate WER per dataset
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import librosa
import websockets
from jiwer import process_words

from normalize import normalize_indic

PULSE_WS_URL = "wss://api.smallest.ai/waves/v1/pulse/get_text"


async def transcribe_one(audio_path: str, api_key: str, language: str = "hi",
                         sample_rate: int = 16000, max_retries: int = 3) -> str:
    """Stream a single audio through Pulse STT and return the joined transcript.

    Raises RuntimeError on permanent failure after retries.
    """
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    chunk_samples = int(0.160 * sample_rate)
    pcm = (audio * 32768.0).astype(np.int16).tobytes()

    params = {"language": language, "encoding": "linear16", "sample_rate": sample_rate}
    ws_url = f"{PULSE_WS_URL}?{urlencode(params)}"

    for attempt in range(max_retries + 1):
        segments: list[str] = []
        received_is_last = False
        try:
            async with websockets.connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                open_timeout=15,
            ) as ws:
                # Send audio in 160 ms chunks
                for offset in range(0, len(audio), chunk_samples):
                    chunk = audio[offset:offset + chunk_samples]
                    await ws.send((chunk * 32768.0).astype(np.int16).tobytes())
                await ws.send(json.dumps({"type": "finalize"}))

                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(message)
                    except asyncio.TimeoutError:
                        break

                    if data.get("error"):
                        raise RuntimeError(f"server error: {data.get('error')}")
                    if data.get("is_final"):
                        t = data.get("transcript")
                        if isinstance(t, str):
                            segments.append(t)
                    if data.get("is_last"):
                        received_is_last = True
                        break

            if not received_is_last and not segments:
                raise RuntimeError("session ended without segments (server stall)")
            return "".join(segments).strip()

        except (websockets.exceptions.WebSocketException, RuntimeError, OSError) as e:
            if attempt < max_retries:
                await asyncio.sleep(1.5 * (2 ** attempt))
                continue
            raise RuntimeError(f"Pulse failed after {max_retries+1} attempts on {audio_path}: {e}") from e


async def transcribe_dataset(dataset_dir: Path, api_key: str, language: str,
                              concurrency: int) -> dict:
    """Transcribe all audios in a dataset directory; write pulse_hyps.jsonl."""
    manifest_rows = [json.loads(l) for l in (dataset_dir / "manifest.jsonl").open()]
    print(f"  {dataset_dir.relative_to(dataset_dir.parents[1])}: {len(manifest_rows)} audios", flush=True)

    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = [None] * len(manifest_rows)
    errors = 0
    done_count = [0]
    t0 = time.time()

    async def worker(i: int, row: dict):
        nonlocal errors
        ap = row["audio_filepath"]
        # Resolve relative path against the dataset dir
        if not Path(ap).is_absolute():
            ap = str(dataset_dir / ap)
        async with sem:
            try:
                hyp = await transcribe_one(ap, api_key=api_key, language=language)
            except Exception as e:
                errors += 1
                hyp = ""
                print(f"    [err] {ap}: {e}", file=sys.stderr, flush=True)
            results[i] = {"audio_filepath": row["audio_filepath"], "hypothesis": hyp}
            done_count[0] += 1
            if done_count[0] % 50 == 0:
                rate = done_count[0] / max(time.time() - t0, 1e-9)
                eta = (len(manifest_rows) - done_count[0]) / max(rate, 1e-9)
                print(f"    {done_count[0]}/{len(manifest_rows)}  "
                      f"({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

    await asyncio.gather(*(worker(i, r) for i, r in enumerate(manifest_rows)))

    # write pulse_hyps.jsonl line-aligned with manifest
    out = dataset_dir / "pulse_hyps.jsonl"
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "dataset": str(dataset_dir.relative_to(dataset_dir.parents[1])),
        "n": len(manifest_rows),
        "errors": errors,
        "wall_s": round(time.time() - t0, 1),
    }


def compute_wer(dataset_dir: Path) -> dict:
    """Compute WER for Pulse, Sarvam, and (if present) Our hypotheses, all
    against the manifest reference, after normalize_indic."""
    mani = [json.loads(l) for l in (dataset_dir / "manifest.jsonl").open()]
    pulse_rows = [json.loads(l) for l in (dataset_dir / "pulse_hyps.jsonl").open()]
    sar_rows = [json.loads(l) for l in (dataset_dir / "sarvam_hyps.jsonl").open()]
    our_rows = [json.loads(l) for l in (dataset_dir / "our_hyps.jsonl").open()] \
        if (dataset_dir / "our_hyps.jsonl").exists() else None

    refs = [normalize_indic(m["text"]) for m in mani]
    pulse_n = [normalize_indic(r["hypothesis"]) for r in pulse_rows]
    sar_n = [normalize_indic(r["hypothesis"]) for r in sar_rows]
    our_n = [normalize_indic(r["hypothesis"]) for r in our_rows] if our_rows else None

    def wer(refs, hyps):
        v = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
        if not v: return None
        rs, hs = zip(*v)
        return process_words(list(rs), list(hs)).wer * 100, len(v)

    pw = wer(refs, pulse_n)
    sw = wer(refs, sar_n)
    ow = wer(refs, our_n) if our_n else None
    return {
        "dataset": str(dataset_dir.relative_to(dataset_dir.parents[1])),
        "n": pw[1] if pw else 0,
        "pulse_wer": round(pw[0], 2) if pw else None,
        "sarvam_wer": round(sw[0], 2) if sw else None,
        "our_wer_in_pack": round(ow[0], 2) if ow else None,
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack-dir", type=Path, default=Path("evals_acefone_test"),
                    help="path to extracted evals_acefone_test/ folder")
    ap.add_argument("--api-key", default=os.environ.get("PULSE_API_KEY"),
                    help="Pulse STT API key (or set PULSE_API_KEY env var)")
    ap.add_argument("--language", default="hi", help="ASR language code")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="concurrent websocket sessions (default 8)")
    ap.add_argument("--only", default=None,
                    help="run only specific dataset (e.g. categories/accent, hinglish/coshe500)")
    ap.add_argument("--skip-transcribe", action="store_true",
                    help="skip transcription, only re-score (assumes pulse_hyps.jsonl exists)")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("ERROR: Pulse API key required. Pass --api-key or set PULSE_API_KEY.")
    if not args.pack_dir.exists():
        sys.exit(f"ERROR: pack dir not found: {args.pack_dir}")

    # Walk all dataset dirs (categories / benchmarks / hinglish)
    dataset_dirs = []
    for section in ("categories", "benchmarks", "hinglish"):
        sec = args.pack_dir / section
        if not sec.exists(): continue
        for d in sorted(sec.iterdir()):
            if d.is_dir() and (d / "manifest.jsonl").exists():
                rel = f"{section}/{d.name}"
                if args.only is None or args.only == rel or args.only == section:
                    dataset_dirs.append(d)

    print(f"Found {len(dataset_dirs)} datasets to process")
    if not dataset_dirs: sys.exit("Nothing to do.")
    print(f"Language: {args.language}  |  Concurrency: {args.concurrency}\n")

    # ── Transcribe pass ──
    if not args.skip_transcribe:
        print("=== Transcription pass ===")
        for d in dataset_dirs:
            r = await transcribe_dataset(d, api_key=args.api_key,
                                          language=args.language,
                                          concurrency=args.concurrency)
            print(f"  ✓ {r['dataset']}: n={r['n']} errors={r['errors']} {r['wall_s']}s\n")

    # ── Scoring pass ──
    print("=== Scoring pass ===")
    summary = []
    print(f"{'Dataset':<35}  {'N':>5}  {'Pulse':>7}  {'Sarvam':>7}  {'Δ Pulse-Sar':>12}")
    print("─"*80)
    for d in dataset_dirs:
        if not (d / "pulse_hyps.jsonl").exists():
            print(f"  {d.relative_to(args.pack_dir)}: no pulse_hyps.jsonl, skipping")
            continue
        r = compute_wer(d)
        summary.append(r)
        pw = r["pulse_wer"]; sw = r["sarvam_wer"]
        delta = (pw - sw) if (pw is not None and sw is not None) else None
        pw_s = f"{pw:.2f}%" if pw is not None else "—"
        sw_s = f"{sw:.2f}%" if sw is not None else "—"
        d_s = f"{delta:+.2f}" if delta is not None else "—"
        print(f"  {r['dataset']:<33}  {r['n']:>5}  {pw_s:>7}  {sw_s:>7}  {d_s:>12}")

    out = args.pack_dir / "pulse_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✓ summary saved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
