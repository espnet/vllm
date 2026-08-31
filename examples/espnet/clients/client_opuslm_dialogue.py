# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference client for an OpusLM-dialogue vLLM server.

Tasks:
  audio_dialogue user turn is audio; the model answers (task token 89)
  text_dialogue  user turn is text; the model answers (task token 88)

Contract notes:
  - The dialogue prompt is built server-side from the STRUCTURED message
    list in mm_processor_kwargs["messages"] (roles system/user/assistant).
    The LAST message must be an empty-content assistant message: it marks
    the generation target.
  - Audio parts inside mm_processor_kwargs.messages are positional
    references ({"type": "input_audio"} markers): the actual audio data
    travels in the OUTER OpenAI `messages` as standard input_audio
    content parts, in the same order.
  - An optional system message with audio is the speaker prompt
    (padded/trimmed to speaker_prompt_length server-side).
  - `mode` is sent in BOTH mm_processor_kwargs and vllm_xargs.

Usage:
  python client_opuslm_dialogue.py --task audio_dialogue \
      --audio /path/to/user_turn.wav --out out.wav
  python client_opuslm_dialogue.py --task text_dialogue \
      --text "How are you today?" --out out.wav
  # With a speaker prompt for the assistant voice:
  python client_opuslm_dialogue.py --task audio_dialogue \
      --speaker-audio /path/to/speaker.wav --audio /path/to/user_turn.wav
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
    mode = args.task  # "audio_dialogue" or "text_dialogue"

    # Outer OpenAI messages: carry the actual audio bytes (and text).
    # Structured dialogue messages (mm_processor_kwargs["messages"]):
    # mirror the outer list; audio parts are positional markers only.
    outer_messages: list[dict] = []
    dialogue_messages: list[dict] = []

    if args.speaker_audio:
        outer_messages.append(
            {
                "role": "system",
                "content": [audio_content_part(args.speaker_audio)],
            }
        )
        dialogue_messages.append(
            {"role": "system", "content": [{"type": "input_audio"}]}
        )

    if args.task == "audio_dialogue":
        if not args.audio:
            sys.exit("--audio is required for --task audio_dialogue")
        for path in args.audio:
            outer_messages.append(
                {"role": "user", "content": [audio_content_part(path)]}
            )
            dialogue_messages.append(
                {"role": "user", "content": [{"type": "input_audio"}]}
            )
    else:  # text_dialogue
        if not args.text:
            sys.exit("--text is required for --task text_dialogue")
        for text in args.text:
            outer_messages.append({"role": "user", "content": text})
            dialogue_messages.append({"role": "user", "content": text})

    # The final empty assistant message marks the generation target.
    # It only needs to appear in the structured list; the outer messages
    # end with the last user turn.
    dialogue_messages.append({"role": "assistant", "content": ""})

    vllm_xargs = {"mode": mode}
    if args.audio_temperature is not None:
        vllm_xargs["audio_temperature"] = args.audio_temperature
    if args.audio_topk is not None:
        vllm_xargs["audio_topk"] = args.audio_topk
    if args.audio_minlen is not None:
        vllm_xargs["audio_minlen"] = args.audio_minlen
    if args.text_minlen is not None:
        vllm_xargs["text_minlen"] = args.text_minlen

    return {
        "model": args.model,
        "messages": outer_messages,
        "max_tokens": args.max_tokens,
        "mm_processor_kwargs": {"mode": mode, "messages": dialogue_messages},
        "vllm_xargs": vllm_xargs,
    }


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
    parser = argparse.ArgumentParser(
        description="OpusLM-dialogue reference client"
    )
    parser.add_argument(
        "--task", required=True, choices=["audio_dialogue", "text_dialogue"]
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9813)
    parser.add_argument("--model", default="opuslm_dialogue")
    parser.add_argument(
        "--speaker-audio",
        default=None,
        help="Speaker-prompt audio for the system message (optional)",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=None,
        help="User-turn audio file; repeatable (audio_dialogue)",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="User-turn text; repeatable (text_dialogue)",
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
    parser.add_argument(
        "--text-minlen",
        type=int,
        default=None,
        help="Minimum text steps before eos; omitted -> server config default",
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
