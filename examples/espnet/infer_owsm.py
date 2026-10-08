# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OWSM v4/v3.x autoregressive audio generation with vLLM."""

import argparse

import soundfile as sf

from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt
from vllm.tokenizers.owsm import resolve_owsm_language_symbol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="export_owsm.py output directory"
    )
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument(
        "--language", default="eng", help="Checkpoint language symbol, ISO code or name"
    )
    parser.add_argument(
        "--task", default="asr", help="Checkpoint task symbol, e.g. asr"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32"
    )
    parser.add_argument(
        "--attention-backend",
        choices=["TRITON_ATTN", "FLASH_ATTN"],
        default="TRITON_ATTN",
    )
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        tokenizer_mode="owsm",
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        tensor_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"audio": 1},
    )
    tokenizer = llm.get_tokenizer()
    vocab = tokenizer.get_vocab()
    language = resolve_owsm_language_symbol(vocab, args.language)
    task = args.task.removeprefix("<").removesuffix(">")
    task = (
        resolve_owsm_language_symbol(vocab, task[3:], translation=True)
        if task.startswith("st_")
        else f"<{task}>"
    )
    symbols = ["<sos>", language, task, "<notimestamps>"]
    missing = [symbol for symbol in symbols if symbol not in tokenizer.get_vocab()]
    if missing:
        raise ValueError(f"Checkpoint does not define requested symbols: {missing}")
    prefix = [tokenizer.convert_tokens_to_ids(symbol) for symbol in symbols]
    prompts = []
    for audio_path in args.audio:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim != 1:
            raise ValueError("Convert audio to mono before inference")
        prompts.append(
            ExplicitEncoderDecoderPrompt(
                encoder_prompt={
                    "prompt_token_ids": [0],
                    "multi_modal_data": {"audio": (audio, sample_rate)},
                },
                decoder_prompt={"prompt_token_ids": prefix},
            )
        )
    # No-timestamps decoding suppresses the native timestamp vocabulary.
    allowed = [
        i
        for token, i in tokenizer.get_vocab().items()
        if not (
            token.startswith("<")
            and token.endswith(">")
            and token[1:-1].replace(".", "", 1).isdigit()
        )
    ]
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            allowed_token_ids=allowed,
            logprobs=1,
        ),
    )
    for path, result in zip(args.audio, outputs):
        print(
            path,
            tokenizer.decode(result.outputs[0].token_ids, skip_special_tokens=True),
        )


if __name__ == "__main__":
    main()
