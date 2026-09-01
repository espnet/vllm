# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference client for a Bagpiper vLLM server.

Tasks:
  text             plain text generation (stops at <|eot|>, id 3)
  audio_understand audio question answering (input_audio content part)
  tts              text_audio generation (text segment + audio segment)
  tts_cfg          text_audio generation with classifier-free guidance

Contract notes (see the API docs for the full picture):
  - For tts/tts_cfg the top-level `temperature` must equal
    `audio_temperature`: the vLLM sampler runs at the top-level
    temperature for the whole request, and the model pre-multiplies
    text-phase logits by audio_temperature/text_temperature so the
    effective text temperature is text_temperature.
  - tts/tts_cfg send NO stop_token_ids: the phase machine forces eos(2)
    after the audio segment finishes.
  - CFG doubles per-request KV usage (a shadow request is created
    server-side); cfg=1.0 is equivalent to plain tts.
  - The model chooses its own output mode. In text_audio mode it first
    emits a <think> reasoning block, then a text segment describing the
    audio it is about to render, and only then the codec frames. It may
    instead stop after the text at eos(2) without ever emitting eot(3),
    in which case the request legitimately returns no audio.
  - A system message decides whether that happens. Measured on the
    released checkpoint, same prompt and sampling, max_tokens 12000:
    "You are a helpful assistant." gave audio in 7 of 8 requests, and
    omitting the system message gave audio in 0 of 8. So tts/tts_cfg
    default --system to that sentence, which is also what the upstream
    reference client sends on every request. Prompt wording is a much
    weaker lever: without a system message, four different prompt shapes
    (instruction+sentence, bare sentence, "Say: ...", and a sound
    description) all returned text only, 0 of 16.

Usage:
  python client_bagpiper.py --task text --prompt "What is 2+2?"
  python client_bagpiper.py --task audio_understand \
      --audio /path/to/audio.wav --prompt "What sound is in this audio?"
  python client_bagpiper.py --task tts \
      --prompt "Read this aloud in a calm voice: The quick brown fox \
jumps over the lazy dog." --out out.wav
  python client_bagpiper.py --task tts_cfg --cfg 3.0 \
      --prompt "A dog barking twice in a quiet room." --out out.wav
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
    """Build an input_audio content part from a local audio file.

    The raw file bytes are base64-encoded; the format field is the file
    suffix (e.g. "wav", "flac").
    """
    with open(path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    fmt = Path(path).suffix.lstrip(".")
    return {
        "type": "input_audio",
        "input_audio": {"data": audio_b64, "format": fmt},
    }


def build_payload(args) -> dict:
    if args.task == "text":
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        return {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.text_temperature,
            "top_k": args.top_k,
            # Stop at <|eot|> (3); eos(2) is the engine default.
            "stop_token_ids": [3],
        }

    if args.task == "audio_understand":
        if not args.audio:
            sys.exit("--audio is required for --task audio_understand")
        messages = [
            {
                "role": "system",
                "content": args.system
                or "You are an audio understanding assistant.",
            },
            {"role": "user", "content": [audio_content_part(args.audio)]},
            {"role": "user", "content": args.prompt},
        ]
        return {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.text_temperature,
            "stop_token_ids": [3],
        }

    if args.task in ("tts", "tts_cfg"):
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        vllm_xargs = {
            "mode": "text_audio",
            "phase": "text",
            "text_temperature": args.text_temperature,
            "audio_temperature": args.audio_temperature,
            "audio_topk": args.audio_topk,
        }
        if args.task == "tts_cfg":
            vllm_xargs["cfg"] = args.cfg
        return {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            # Top-level temperature MUST equal audio_temperature; the
            # model compensates text-phase logits internally.
            "temperature": args.audio_temperature,
            "top_k": args.audio_topk,
            # No stop_token_ids: the phase machine forces eos(2) itself.
            "vllm_xargs": vllm_xargs,
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
    parser = argparse.ArgumentParser(description="Bagpiper reference client")
    parser.add_argument(
        "--task",
        required=True,
        choices=["text", "audio_understand", "tts", "tts_cfg"],
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9811)
    parser.add_argument("--model", default="bagpiper")
    # The default carries the sentence to render, not just the instruction,
    # so that the rendered audio is checkable against a known target.
    parser.add_argument(
        "--prompt",
        default=(
            "Read this aloud in a calm voice: "
            "The quick brown fox jumps over the lazy dog."
        ),
    )
    parser.add_argument(
        "--system",
        default=None,
        help=(
            "System message. Default: 'You are a helpful assistant.' for "
            "tts/tts_cfg (it is what makes the model emit audio at all), "
            "none otherwise. Pass --system '' to send none."
        ),
    )
    parser.add_argument(
        "--audio", default=None, help="Audio file (audio_understand)"
    )
    parser.add_argument("--out", default="out.wav", help="Output WAV path")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Default: 12000 for tts/tts_cfg, 4096 otherwise",
    )
    parser.add_argument("--text-temperature", type=float, default=0.6)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-topk", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    if args.max_tokens is None:
        args.max_tokens = 12000 if args.task in ("tts", "tts_cfg") else 4096

    # The system message is what decides whether tts returns audio at all
    # (7/8 with it, 0/8 without -- see the contract notes above), so default
    # it on for the two audio tasks. `--system ''` still opts out.
    if args.system is None and args.task in ("tts", "tts_cfg"):
        args.system = "You are a helpful assistant."

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = build_payload(args)

    resp = requests.post(url, json=payload, timeout=args.timeout)
    if resp.status_code != 200:
        sys.exit(f"HTTP {resp.status_code}: {resp.text[:2000]}")

    handle_response(resp.json(), args.out)


if __name__ == "__main__":
    main()
