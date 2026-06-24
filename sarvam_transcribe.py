#!/usr/bin/env python3
"""Transcribe a manifest with Sarvam saaras:v3 streaming.

Reads a JSONL manifest (one `{"audio_filepath": "...", ...}` object per line),
streams each audio file to Sarvam's WebSocket in codemix mode, and writes
`{"audio_filepath": "...", "hypothesis": "..."}` for every successful
transcription. Resumable — re-running picks up where the last run stopped.

This is the same script that produced the bundled `sarvam_hyps.jsonl` files
in the eval pack.

Setup:
    pip install -r requirements.txt
    pip install certifi
    export SARVAM_API_KEY=sk_xxx

Usage:
    python sarvam_transcribe.py \
        --manifest evals_test/categories/accent/manifest.jsonl \
        --audio-root evals_test/categories/accent \
        --output evals_test/categories/accent/sarvam_hyps_local.jsonl \
        --cache-dir .sarvam_cache/accent
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import ssl
import sys
import time
import urllib.parse
from pathlib import Path

import certifi
import websockets

SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"
LANGUAGE_CODE = "hi-IN"
MODEL = "saaras:v3"
MODE = "codemix"
SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 1.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("sarvam")

STOP = asyncio.Event()


def cache_key(audio_path: str) -> str:
    h = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    return f"{Path(audio_path).stem}__{h}"


def cached_path(cache_dir: Path, audio_path: str) -> Path:
    return cache_dir / f"{cache_key(audio_path)}.json"


def load_audio_pcm16(audio_path: str) -> bytes:
    import soundfile as sf
    data, sr = sf.read(audio_path, dtype="int16")
    if data.ndim > 1:
        import numpy as np
        data = data.mean(axis=1).astype("int16")
    if sr != SAMPLE_RATE:
        import librosa
        data_f = data.astype("float32") / 32768.0
        data_f = librosa.resample(data_f, orig_sr=sr, target_sr=SAMPLE_RATE)
        data = (data_f * 32767).clip(-32768, 32767).astype("int16")
    return data.tobytes()


async def transcribe_one(audio_path: str, api_key: str, max_retries: int = 3) -> tuple[str, str]:
    try:
        raw = load_audio_pcm16(audio_path)
    except Exception as e:
        return "", f"read_failed: {type(e).__name__}: {e}"
    if not raw:
        return "", "no_audio"

    params = {
        "language-code": LANGUAGE_CODE,
        "model": MODEL,
        "sample_rate": str(SAMPLE_RATE),
        "input_audio_codec": "pcm_s16le",
        "mode": MODE,
    }
    ws_url = f"{SARVAM_WS_URL}?{urllib.parse.urlencode(params)}"
    headers = {"Api-Subscription-Key": api_key}
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    chunk_bytes = int(CHUNK_DURATION_SEC * SAMPLE_RATE) * 2

    last_err = None
    for attempt in range(max_retries + 1):
        if STOP.is_set():
            return "", "aborted"
        try:
            async with websockets.connect(
                ws_url, additional_headers=headers, ssl=ssl_ctx,
                close_timeout=30, max_size=2**20,
                ping_interval=None, ping_timeout=None,
            ) as ws:
                finals: list[str] = []
                for off in range(0, len(raw), chunk_bytes):
                    chunk = raw[off:off + chunk_bytes]
                    msg = json.dumps({
                        "audio": {
                            "data": base64.b64encode(chunk).decode("ascii"),
                            "sample_rate": str(SAMPLE_RATE),
                            "encoding": "audio/wav",
                        },
                    })
                    await ws.send(msg)
                    await asyncio.sleep(CHUNK_DURATION_SEC)
                await ws.send(json.dumps({"type": "flush"}))

                try:
                    while True:
                        rmsg = await asyncio.wait_for(ws.recv(), timeout=15)
                        data = json.loads(rmsg)
                        t = data.get("type", "")
                        if t == "error":
                            err = data.get("data", {}).get("message", json.dumps(data))
                            raise RuntimeError(f"sarvam_error: {err}")
                        if t == "data":
                            transcript = data.get("data", {}).get("transcript", "")
                            if transcript.strip():
                                finals.append(transcript)
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass

            text = " ".join(finals).strip()
            return text, ("ok" if text else "empty_text")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < max_retries and not STOP.is_set():
                await asyncio.sleep(1.5 * (2 ** attempt))
                continue
            return "", f"failed_after_{max_retries}_retries: {last_err}"
    return "", f"unreachable: {last_err}"


async def main_async(args):
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        sys.exit("ERROR: SARVAM_API_KEY env var not set.")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fail_log = args.output.with_suffix(args.output.suffix + ".failures")

    logger.info(f"manifest:    {args.manifest}")
    logger.info(f"audio-root:  {args.audio_root}")
    logger.info(f"output:      {args.output}")
    logger.info(f"cache:       {args.cache_dir}")
    logger.info(f"mode={MODE} model={MODEL} lang={LANGUAGE_CODE} conc={args.concurrency}")

    audios: list[tuple[str, str]] = []   # (manifest_path, resolved_disk_path)
    seen = set()
    with args.manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            ap = e["audio_filepath"]
            if ap in seen: continue
            seen.add(ap)
            disk_path = str((args.audio_root / ap).resolve()) if args.audio_root else ap
            audios.append((ap, disk_path))
    if args.limit:
        audios = audios[:args.limit]
    logger.info(f"loaded {len(audios)} unique audios")

    done_set = set()
    if args.output.exists():
        with args.output.open() as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get(args.field_name):
                        done_set.add(e["audio_filepath"])
                except Exception:
                    pass
    pending = [(ap, dp) for (ap, dp) in audios if ap not in done_set]
    logger.info(f"resume: {len(done_set)} done, {len(pending)} pending")

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    out_f = args.output.open("a", encoding="utf-8")
    fail_f = fail_log.open("a", encoding="utf-8")
    t_start = time.time()
    counter = {"ok": 0, "err": 0, "done": 0}

    async def process(ap, dp):
        async with sem:
            cp = cached_path(args.cache_dir, ap)
            text = ""
            note = ""
            if cp.exists():
                try:
                    cd = json.loads(cp.read_text())
                    text = cd.get("transcript", "") or ""
                    note = "cache_hit" if text else "cache_empty"
                except Exception:
                    pass
            if not text:
                text, note = await transcribe_one(dp, api_key, args.max_retries)
                try:
                    cp.write_text(json.dumps({"transcript": text, "note": note}, ensure_ascii=False))
                except Exception:
                    pass

            async with out_lock:
                if text:
                    counter["ok"] += 1
                    out_f.write(json.dumps({"audio_filepath": ap, args.field_name: text}, ensure_ascii=False) + "\n")
                    out_f.flush()
                else:
                    counter["err"] += 1
                    fail_f.write(f"{ap}\t{note}\n")
                    fail_f.flush()
                counter["done"] += 1
                if counter["done"] % args.log_every == 0:
                    rate = counter["done"] / max(time.time() - t_start, 1e-9)
                    eta_h = (len(pending) - counter["done"]) / max(rate, 1e-9) / 3600
                    logger.info(f"progress: {counter['done']}/{len(pending)} | "
                                f"ok={counter['ok']} err={counter['err']} | "
                                f"{rate:.1f}/s | ETA {eta_h:.2f}h")

    try:
        await asyncio.gather(*(process(ap, dp) for (ap, dp) in pending))
    finally:
        out_f.close(); fail_f.close()

    logger.info(f"DONE in {time.time() - t_start:.1f}s — ok={counter['ok']} err={counter['err']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True,
                    help="JSONL manifest with `audio_filepath` per line.")
    ap.add_argument("--audio-root", type=Path, default=None,
                    help="Directory the manifest paths are relative to (e.g. the dataset dir in the pack).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output JSONL — one `{audio_filepath, hypothesis}` per line.")
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="Per-file cache dir (for resume).")
    ap.add_argument("--field-name", default="hypothesis",
                    help="Field name for the transcript in the output JSONL (default: hypothesis, matches bundled sarvam_hyps.jsonl).")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
