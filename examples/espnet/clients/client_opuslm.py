# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference client for an OpusLM vLLM server.

Tasks:
  tts    text -> speech (plain TTS, task token 82; with an input_audio
         voice prompt the server resolves to voice-cloning TTS, 81)
  asr    speech -> text (task token 80)
  textlm plain text LM continuation (task token 64)

Contract notes:
  - `mode` is sent in BOTH mm_processor_kwargs (prompt-build time) and
    vllm_xargs (decode time). The processor uses it for the prompt
    layout; the runner uses it for the phase machine.
  - OpusLM has no CFG.
  - eos is 5 (from hf_config); no stop_token_ids are needed.
  - For TTS, validate on message.audio.data. The transcript/content
    re-derivation is config-driven after the migration but the audio
    payload is the source of truth.

Usage:
  python client_opuslm.py --task tts --prompt "Hello world" --out out.wav
  python client_opuslm.py --task asr --audio /path/to/audio.wav
  python client_opuslm.py --task textlm --prompt "Once upon a time"
"""

import argparse
import base64
import io
import json
import sys
import wave
from pathlib import Path

import requests


def audio_content_part(path: str) -> dict:
    """Build an input_audio content part from a local audio file."""
    with open(path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    fmt = Path(path).suffix.lstrip(".")
    return {
        "type": "input_audio",
        "input_audio": {"data": audio_b64, "format": fmt},
    }


def build_payload(args) -> dict:
    if args.task == "tts":
        mode = "text_audio"
        messages = [{"role": "user", "content": args.prompt}]
        if args.audio:
            # Optional voice-cloning prompt: with audio input present the
            # server resolves text_audio to TTS (81) instead of plain TTS.
            messages.insert(
                0, {"role": "user", "content": [audio_content_part(args.audio)]}
            )
        vllm_xargs = {
            "mode": mode,
            "audio_temperature": args.audio_temperature,
            "audio_topk": args.audio_topk,
        }
        # audio_minlen: omit unless given; the server falls back to the
        # config default (50 for OpusLM).
        if args.audio_minlen is not None:
            vllm_xargs["audio_minlen"] = args.audio_minlen
        return {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "mm_processor_kwargs": {"mode": mode},
            "vllm_xargs": vllm_xargs,
        }

    if args.task == "asr":
        if not args.audio:
            sys.exit("--audio is required for --task asr")
        mode = "audio_text"
        messages = [
            {"role": "user", "content": [audio_content_part(args.audio)]}
        ]
        return {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "mm_processor_kwargs": {"mode": mode},
            "vllm_xargs": {"mode": mode},
        }

    if args.task == "textlm":
        mode = "text_text"
        messages = [{"role": "user", "content": args.prompt}]
        return {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "mm_processor_kwargs": {"mode": mode},
            "vllm_xargs": {"mode": mode},
        }

    raise ValueError(f"Unknown task: {args.task}")


def handle_response(data: dict, out_path: str) -> None:
    choice = data["choices"][0]
    message = choice["message"]

    print(f"content: {message.get('content', '')!r}")
    print(f"finish_reason: {choice.get('finish_reason')}")
    print(f"usage: {json.dumps(data.get('usage'))}")

    audio = message.get("audio")
    if audio and audio.get("data"):
        wav_bytes = base64.b64decode(audio["data"])
        # Validate: the payload must be a well-formed RIFF/WAV.
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
            print(
                f"audio: {duration:.2f}s @ {wf.getframerate()} Hz, "
                f"{wf.getnchannels()} ch"
            )
        Path(out_path).write_bytes(wav_bytes)
        print(f"saved audio to {out_path}")
    else:
        print("audio: (none)")


def main():
    parser = argparse.ArgumentParser(description="OpusLM reference client")
    parser.add_argument(
        "--task", required=True, choices=["tts", "asr", "textlm"]
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9812)
    parser.add_argument("--model", default="opuslm")
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument(
        "--audio",
        default=None,
        help="Audio file (required for asr; optional voice prompt for tts)",
    )
    parser.add_argument("--out", default="out.wav", help="Output WAV path")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-topk", type=int, default=30)
    parser.add_argument(
        "--audio-minlen",
        type=int,
        default=None,
        help="Minimum audio steps before eos; omitted -> server config default",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = build_payload(args)

    resp = requests.post(url, json=payload, timeout=args.timeout)
    if resp.status_code != 200:
        sys.exit(f"HTTP {resp.status_code}: {resp.text[:2000]}")

    handle_response(resp.json(), args.out)


if __name__ == "__main__":
    main()
