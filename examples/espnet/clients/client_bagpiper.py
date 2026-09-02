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

BAGPIPER IS NOT A TEXT-TO-SPEECH ENGINE. Read this before writing a prompt.

  It is a descriptive audio generator. The training data (the `system` and
  `user` turns of filtered_realistic.jsonl) phrases every request as a
  natural-language description of a scene, with any speech content quoted
  inside that description:

      "A clear female voice explains how to change the size of photo copies
       when exporting from Adobe Lightroom. She speaks slowly and precisely,
       with a neutral, professional tone. After she finishes saying '...each
       resizing export option does,' a short electronic buzzing sound plays
       and then cuts off suddenly."

  A "read this aloud: <sentence>" instruction is NOT a shape the model was
  trained on. It still returns audio, but the audio is loosely conditioned
  and is not a faithful reading of the sentence. To make the model say a
  specific line, quote the line inside a scene description -- see the
  DEFAULT_TTS_PROMPT below, which is a validated example.

  The system message is not free either. Every sampled entry of the training
  data carries one constant system prompt, DEFAULT_TTS_SYSTEM below (364
  chars, md5 903599715f6bea955adc1cfbf83aa9e2). It is what tells the model to
  think first and then describe the audio, which is exactly the output
  contract above.

  A previous revision of this file defaulted --system to "You are a helpful
  assistant." and claimed that string was "what the upstream reference client
  sends on every request". That claim was wrong. In the upstream bundle the
  string occurs exactly once, inside a `#` comment block at
  scripts/serve_cfg_1.sh:17 (an abbreviated curl doc example). The runnable
  reference client, scripts/client_all.py, hardcodes no system turn at all --
  it takes the system turn from the dataset, where it is always the long
  prompt below.

  Validated 2026-09-02 on an H100 against the release checkpoint
  (speechlm-qwen3-8b, shard-4 md5 0ec3d02b31ada2f9e89078d89a7d34b3) using the
  upstream runtime: 5 of 5 requests returned audio, all finish_reason=stop,
  no length truncation, and where the quoted span was the complete utterance
  the speaking rates were 2.78/3.06/2.84 words per second -- natural. Evidence
  in terminal_docs/audio_samples/bagpiper_authoritative_validation/.

Usage:
  python client_bagpiper.py --task text --prompt "What is 2+2?"
  python client_bagpiper.py --task audio_understand \
      --audio /path/to/audio.wav --prompt "What sound is in this audio?"
  # tts/tts_cfg: describe the scene and quote what should be said.
  python client_bagpiper.py --task tts --out out.wav
  python client_bagpiper.py --task tts --out out.wav \
      --prompt "A calm male voice, close-miked in a quiet studio, says: \
'Your package will arrive on Tuesday.' No background noise."
  python client_bagpiper.py --task tts_cfg --cfg 3.0 --out out.wav
"""

import argparse
import base64
import io
import json
import sys
import wave
from pathlib import Path

import requests

# The one system prompt that every sampled entry of the Bagpiper SFT data
# carries, verbatim (364 chars, md5 903599715f6bea955adc1cfbf83aa9e2). Source:
# the `system` turn of bagpiper_sft/sft_part2/filtered_realistic.jsonl, constant
# across 4000 sampled entries. Do not paraphrase it -- it is what puts the model
# into think-then-describe-then-render mode.
DEFAULT_TTS_SYSTEM = (
    "You are a helpful assistant that generates audio based on user requests. "
    "You can create various types of audio including sound effects, music, "
    "speech, ambient sounds, and any combination of these. When given a "
    "request, first think through what the user wants and how to create "
    "high-quality audio, then provide a detailed description of the audio you "
    "will generate."
)

# An in-distribution default: a scene description with the spoken line quoted
# inside it. Validated 2026-09-02 -- 1.80 s of audio for this sentence,
# finish_reason=stop, 2.78 words/sec.
DEFAULT_TTS_PROMPT = (
    "A clear, friendly female voice, close-miked in a quiet room, says: "
    "'Hello, how are you today?'. She speaks at a relaxed, natural pace with "
    "a warm tone and no background noise or music."
)


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
        default=DEFAULT_TTS_PROMPT,
    )
    parser.add_argument(
        "--system",
        default=None,
        help=(
            "System message. Default for tts/tts_cfg: DEFAULT_TTS_SYSTEM, the "
            "constant audio-generation system prompt from the training data "
            "(see the module docstring). It is what puts the model into "
            "think-then-describe-then-render mode. None for the other tasks. "
            "Pass --system '' to send none."
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
        args.system = DEFAULT_TTS_SYSTEM

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = build_payload(args)

    resp = requests.post(url, json=payload, timeout=args.timeout)
    if resp.status_code != 200:
        sys.exit(f"HTTP {resp.status_code}: {resp.text[:2000]}")

    handle_response(resp.json(), args.out)


if __name__ == "__main__":
    main()
