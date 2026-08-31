# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined stress test client for Bagpiper: sends text-only, audiogen, and
audiogen+CFG requests ALL concurrently to hammer the vLLM server.

Each task type uses 128 concurrent connections (384 total).
All results saved to the output dir under separate files.

Input JSONL (ESPnet triplet format), one example per line:
    {"example_id": ..., "messages": [[role, "text"|"audio", content], ...]}
Audio content is a file path; the client base64-encodes the raw file
bytes and sets format from the file suffix.

Usage:
    python client_bagpiper_stress.py \
        --audiogen-input /path/to/audiogen.jsonl \
        --text-input /path/to/text.jsonl
    python client_bagpiper_stress.py --port 9811 --concurrency 128 ...
"""

import argparse
import asyncio
import base64
import io
import json
import random
import time
import wave
from pathlib import Path

import aiohttp

# ── Defaults ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output_stress"

MAX_TOKENS = 12000
MODEL = "bagpiper"


# ── Message conversion ───────────────────────────────────────────────────────
def convert_messages_audiogen(raw_messages):
    """Convert for audiogen (skip assistant ground truth)."""
    messages = []
    for role, modality, content in raw_messages:
        if role == "assistant":
            continue
        if modality == "text":
            messages.append({"role": role, "content": content})
        elif modality == "audio":
            with open(content, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            fmt = Path(content).suffix.lstrip(".")
            messages.append({
                "role": role,
                "content": [{"type": "input_audio",
                              "input_audio": {"data": audio_b64, "format": fmt}}],
            })
        else:
            raise ValueError(f"Unknown modality: {modality}")
    return messages


def convert_messages_text(raw_messages):
    """Convert for text-only (keep all roles including assistant)."""
    messages = []
    for role, modality, content in raw_messages:
        if modality == "text":
            messages.append({"role": role, "content": content})
        elif modality == "audio":
            with open(content, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            fmt = Path(content).suffix.lstrip(".")
            messages.append({
                "role": role,
                "content": [{"type": "input_audio",
                              "input_audio": {"data": audio_b64, "format": fmt}}],
            })
        else:
            raise ValueError(f"Unknown modality: {modality}")
    return messages


# ── Per-request senders ──────────────────────────────────────────────────────
async def send_audiogen_cfg(session, url, model, eid, msgs, max_tokens,
                            cfg, audio_temp, text_temp, audio_topk, sem):
    """text_audio + CFG request."""
    payload = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": audio_temp,
        "top_k": audio_topk,
        "vllm_xargs": {
            "mode": "text_audio", "phase": "text",
            "text_temperature": text_temp,
            "audio_temperature": audio_temp,
            "audio_topk": audio_topk,
            "cfg": cfg,
        },
    }
    async with sem:
        t0 = time.monotonic()
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                lat = time.monotonic() - t0
                if resp.status != 200:
                    return {"example_id": eid, "error": data.get("message", str(data)),
                            "status": resp.status, "latency": lat}
                if "choices" not in data:
                    return {"example_id": eid, "error": "no choices",
                            "raw_response": data, "latency": lat}
                ch = data["choices"][0]
                msg = ch["message"]
                return {
                    "example_id": eid, "task": "audiogen_cfg",
                    "text": msg.get("content", ""),
                    "audio_base64": (msg["audio"]["data"]
                                     if msg.get("audio") else None),
                    "finish_reason": ch["finish_reason"],
                    "usage": data.get("usage"), "latency": lat,
                }
        except Exception as e:
            return {"example_id": eid, "error": str(e),
                    "latency": time.monotonic() - t0}


async def send_audiogen(session, url, model, eid, msgs, max_tokens,
                        audio_temp, text_temp, audio_topk, sem):
    """text_audio (no CFG) request."""
    payload = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": audio_temp,
        "top_k": audio_topk,
        "vllm_xargs": {
            "mode": "text_audio", "phase": "text",
            "text_temperature": text_temp,
            "audio_temperature": audio_temp,
            "audio_topk": audio_topk,
        },
    }
    async with sem:
        t0 = time.monotonic()
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                lat = time.monotonic() - t0
                if resp.status != 200:
                    return {"example_id": eid, "error": data.get("message", str(data)),
                            "status": resp.status, "latency": lat}
                if "choices" not in data:
                    return {"example_id": eid, "error": "no choices",
                            "raw_response": data, "latency": lat}
                ch = data["choices"][0]
                msg = ch["message"]
                return {
                    "example_id": eid, "task": "audiogen",
                    "text": msg.get("content", ""),
                    "audio_base64": (msg["audio"]["data"]
                                     if msg.get("audio") else None),
                    "finish_reason": ch["finish_reason"],
                    "usage": data.get("usage"), "latency": lat,
                }
        except Exception as e:
            return {"example_id": eid, "error": str(e),
                    "latency": time.monotonic() - t0}


async def send_text(session, url, model, eid, msgs, max_tokens,
                    temperature, top_k, sem):
    """Text-only request (stop on eot)."""
    payload = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "stop_token_ids": [3],
    }
    async with sem:
        t0 = time.monotonic()
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                lat = time.monotonic() - t0
                if resp.status != 200:
                    return {"example_id": eid, "error": data.get("message", str(data)),
                            "status": resp.status, "latency": lat}
                if "choices" not in data:
                    return {"example_id": eid, "error": "no choices",
                            "raw_response": data, "latency": lat}
                ch = data["choices"][0]
                return {
                    "example_id": eid, "task": "text",
                    "response": ch["message"]["content"],
                    "finish_reason": ch["finish_reason"],
                    "usage": data.get("usage"), "latency": lat,
                }
        except Exception as e:
            return {"example_id": eid, "error": str(e),
                    "latency": time.monotonic() - t0}


# ── Stats helper ─────────────────────────────────────────────────────────────
def print_stats(label, results, elapsed):
    n_ok = sum(1 for r in results if "error" not in r)
    n_err = sum(1 for r in results if "error" in r)
    latencies = [r["latency"] for r in results
                 if "latency" in r and "error" not in r]
    total_tok = sum(r.get("usage", {}).get("completion_tokens", 0)
                    for r in results if "usage" in r)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p50 = sorted(latencies)[len(latencies)//2] if latencies else 0
    p99 = sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0

    print(f"\n  [{label}]")
    print(f"    Requests:  {len(results)} ({n_ok} ok, {n_err} err)")
    print(f"    Tokens:    {total_tok}")
    print(f"    Throughput:{total_tok / elapsed:.1f} tok/s")
    print(f"    Latency:   avg={avg_lat:.2f}s  p50={p50:.2f}s  p99={p99:.2f}s")
    if n_err:
        print(f"    First errors:")
        for r in results:
            if "error" in r:
                print(f"      {r['example_id']}: {str(r['error'])[:120]}")
                n_err -= 1
                if n_err <= 5:
                    break


def save_jsonl(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    Saved → {path}")


def save_random_wavs(results, n, save_dir, label):
    """Pick n random results with audio and save as WAV."""
    with_audio = [r for r in results if r.get("audio_base64")]
    if not with_audio:
        print(f"  [{label}] No results with audio to save.")
        return 0
    n = min(n, len(with_audio))
    chosen = random.sample(with_audio, n)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for r in chosen:
        try:
            wav_bytes = base64.b64decode(r["audio_base64"])
            buf = io.BytesIO(wav_bytes)
            with wave.open(buf, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
            eid = r["example_id"]
            fname = f"{eid}_{label}.wav"
            wav_path = Path(save_dir) / fname
            wav_path.write_bytes(wav_bytes)
            text_preview = (r.get("text") or "")[:80]
            print(f"    {fname:55s} {duration:5.2f}s  {text_preview}...")
            saved += 1
        except Exception as e:
            print(f"    WARNING: {r['example_id']}: {e}")
    return saved


# ── Main ─────────────────────────────────────────────────────────────────────
async def main(args):
    url = f"http://localhost:{args.port}/v1/chat/completions"
    conc = args.concurrency  # per-task concurrency
    total_conc = conc * 3

    # Separate semaphores per task type so each gets its own 128 slots
    sem_cfg = asyncio.Semaphore(conc)
    sem_gen = asyncio.Semaphore(conc)
    sem_txt = asyncio.Semaphore(conc)

    # ── Load audiogen data ───────────────────────────────────────────────
    ag_examples = []
    with open(args.audiogen_input) as f:
        for line in f:
            if line.strip():
                ag_examples.append(json.loads(line))
    print(f"[audiogen]     Loaded {len(ag_examples)} examples "
          f"from {args.audiogen_input}")

    ag_converted = []
    for ex in ag_examples:
        msgs = convert_messages_audiogen(ex["messages"])
        ag_converted.append((ex["example_id"], msgs))

    # ── Load text data ───────────────────────────────────────────────────
    txt_examples = []
    with open(args.text_input) as f:
        for line in f:
            if line.strip():
                txt_examples.append(json.loads(line))
    print(f"[text]         Loaded {len(txt_examples)} examples "
          f"from {args.text_input}")

    txt_converted = []
    for ex in txt_examples:
        msgs = convert_messages_text(ex["messages"])
        txt_converted.append((ex["example_id"], msgs))

    # ── Build ALL tasks ──────────────────────────────────────────────────
    print(f"\nLaunching stress test: {conc} concurrency per task, "
          f"{total_conc} total connections")
    print(f"  max_tokens={args.max_tokens}, cfg={args.cfg}")
    print(f"  audiogen_cfg: {len(ag_converted)} requests")
    print(f"  audiogen:     {len(ag_converted)} requests")
    print(f"  text:         {len(txt_converted)} requests")
    total_reqs = len(ag_converted) * 2 + len(txt_converted)
    print(f"  TOTAL:        {total_reqs} requests")

    # ── Fire ALL at once with a shared session ───────────────────────────
    t0 = time.time()
    connector = aiohttp.TCPConnector(limit=total_conc)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=3600),
    ) as session:
        all_tasks = []
        for eid, msgs in ag_converted:
            all_tasks.append(send_audiogen_cfg(
                session, url, args.model, eid, msgs, args.max_tokens,
                args.cfg, args.audio_temperature, args.text_temperature,
                args.audio_topk, sem_cfg))
        n_cfg = len(all_tasks)

        for eid, msgs in ag_converted:
            all_tasks.append(send_audiogen(
                session, url, args.model, eid, msgs, args.max_tokens,
                args.audio_temperature, args.text_temperature,
                args.audio_topk, sem_gen))
        n_gen = len(all_tasks) - n_cfg

        for eid, msgs in txt_converted:
            all_tasks.append(send_text(
                session, url, args.model, eid, msgs, args.max_tokens,
                args.text_temperature, args.top_k, sem_txt))
        n_txt = len(all_tasks) - n_cfg - n_gen

        print(f"\nAll {len(all_tasks)} requests submitted, waiting...")
        all_results = await asyncio.gather(*all_tasks)

    elapsed = time.time() - t0

    # ── Split results back ───────────────────────────────────────────────
    results_cfg = list(all_results[:n_cfg])
    results_gen = list(all_results[n_cfg:n_cfg + n_gen])
    results_txt = list(all_results[n_cfg + n_gen:])

    # ── Save & report ────────────────────────────────────────────────────
    out = Path(args.output_dir)
    print(f"\n{'='*65}")
    print(f"STRESS TEST RESULTS  ({elapsed:.1f}s total)")
    print(f"{'='*65}")

    save_jsonl(results_cfg, str(out / "stress_audiogen_cfg.jsonl"))
    print_stats("audiogen_cfg", results_cfg, elapsed)

    save_jsonl(results_gen, str(out / "stress_audiogen.jsonl"))
    print_stats("audiogen", results_gen, elapsed)

    save_jsonl(results_txt, str(out / "stress_text.jsonl"))
    print_stats("text", results_txt, elapsed)

    # Overall
    total_tok = sum(
        r.get("usage", {}).get("completion_tokens", 0)
        for r in all_results if "usage" in r
    )
    n_ok = sum(1 for r in all_results if "error" not in r)
    n_err = sum(1 for r in all_results if "error" in r)
    print(f"\n{'='*65}")
    print(f"OVERALL")
    print(f"{'='*65}")
    print(f"  Total requests: {len(all_results)} ({n_ok} ok, {n_err} err)")
    print(f"  Total tokens:   {total_tok}")
    print(f"  Wall time:      {elapsed:.1f}s")
    print(f"  Throughput:     {total_tok / elapsed:.1f} tok/s")
    print(f"{'='*65}")

    # ── Save random WAV samples ──────────────────────────────────────────
    if args.save_random > 0:
        audio_dir = str(out / "audio")
        print(f"\nSaving {args.save_random} random WAVs per task to {audio_dir}/")
        n1 = save_random_wavs(results_cfg, args.save_random, audio_dir,
                              "cfg")
        n2 = save_random_wavs(results_gen, args.save_random, audio_dir,
                              "nocfg")
        print(f"  Total WAVs saved: {n1 + n2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined stress test: text + audiogen + audiogen_cfg")
    parser.add_argument("--audiogen-input", required=True,
                        help="ESPnet-format JSONL for audiogen requests")
    parser.add_argument("--text-input", required=True,
                        help="ESPnet-format JSONL for text-only requests")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--port", type=int, default=9811)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--text-temperature", type=float, default=0.6)
    parser.add_argument("--audio-topk", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=128,
                        help="Concurrency PER task type (total = 3x)")
    parser.add_argument("--save-random", type=int, default=10,
                        help="Save N random WAV samples per audio task")
    args = parser.parse_args()
    asyncio.run(main(args))
