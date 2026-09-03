# vLLM + ESPnet audio language models

A fork of [vLLM](https://github.com/vllm-project/vllm) that adds serving support
for three ESPnet speech models: **bagpiper**, **opuslm**, and
**opuslm_dialogue**. Everything else is upstream vLLM, unmodified.

> This is a modified fork, not an official vLLM release. Upstream's own README is
> preserved verbatim at [README.vllm.md](README.vllm.md).

## Versions

|  |  |
| --- | --- |
| Upstream base | vLLM **v0.28.0**, tag commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` (2026-08-23) |
| This branch | `espnet-audio-v0.28.0` |
| Docker base image | `vllm/vllm-openai:v0.28.0` |

The three model implementations are Python-only, so compiled kernels come
straight from the upstream v0.28.0 wheel.

For the exact commit of a checkout, run `git rev-parse HEAD`; for a packaged
snapshot, read the manifest that ships beside the archive. This file
deliberately records no commit count — any number written here would be stale
the moment the commit writing it landed.

## The models

| name | backbone | what it does | official weights |
| --- | --- | --- | --- |
| `bagpiper` | Qwen3-8B + Qwen3-Omni audio tower + Xcodec | audio generation from a described scene; 8 codec streams, optional CFG | [`espnet/bagpiper-tts-sft`](https://huggingface.co/espnet/bagpiper-tts-sft) |
| `opuslm` | OLMo-2-7B | TTS, ASR, text LM | [`espnet/OpusLM_7B_Anneal`](https://huggingface.co/espnet/OpusLM_7B_Anneal) |
| `opuslm_dialogue` | SmolLM2-1.7B | spoken dialogue | [`espnet/multi_turn_SDS_RLAIF`](https://huggingface.co/espnet/multi_turn_SDS_RLAIF) |

Use those exact names for `--served-model-name`, and make sure `config.json`
declares the matching `model_type`.

**None of the published checkpoints is directly vLLM-loadable** — they ship raw
ESPnet weights with no `config.json`, tokenizer or safetensors, so a conversion
step is required. [`examples/espnet/MODELS.md`](examples/espnet/MODELS.md) records
which repo and revision each model comes from, how that was proven, and what
conversion each needs.

## Quickstart

```bash
# 1. fetch the official weights (18 GB of native ESPnet checkpoint)
hf download espnet/bagpiper-tts-sft --local-dir ~/models/bagpiper-tts-sft
sha256sum -c ~/models/bagpiper-tts-sft/SHA256SUMS

# 2. convert to a vLLM-loadable directory. --ref-dir supplies config.json and
#    the tokenizer, which espnet does not publish (see MODELS.md).
python examples/espnet/convert/convert_bagpiper_ckpt.py \
    ~/models/bagpiper-tts-sft ~/models/bagpiper-tts-sft-vllm \
    --ref-dir /path/to/a/bagpiper/config+tokenizer/dir
# add --dry-run first to validate weight coverage without writing 18 GB

# 3. serve (port 9811; extra args pass through to `vllm serve`)
MODEL_PATH=~/models/bagpiper-tts-sft-vllm bash examples/espnet/serve_bagpiper.sh

# 4. request
python examples/espnet/clients/client_bagpiper.py --task tts --out demo.wav
```

The converter validates every tensor against the weight groups the model can
load and exits non-zero listing anything unexpected, so a layout change fails
loudly instead of producing a directory that loads with missing weights.

`opuslm` and `opuslm_dialogue` follow the same three steps with
`convert_opuslm_ckpt.py`, `serve_opuslm.sh` / `serve_opuslm_dialogue.sh`, and
`client_opuslm.py` / `client_opuslm_dialogue.py`.

## Bagpiper takes a scene description, not a sentence to read

This is the one thing that trips people up. Bagpiper is **not** a text-to-speech
engine. Describe the audio you want, and quote any spoken line inside that
description:

```bash
python examples/espnet/clients/client_bagpiper.py --task tts --out demo.wav \
    --prompt "A calm male voice, close-miked in a quiet studio, says: \
'Your package will arrive on Tuesday.' No background noise."
```

Writing `"Read this aloud: <sentence>"` is out of distribution. It still returns
audio, but not a faithful reading of your sentence — that mistake produced a
batch of unintelligible samples before it was caught. The client's default
`--system` is the 364-character audio-generation system prompt the model was
trained with; keep it.

### Listen

Real output from `espnet/bagpiper-tts-sft` converted with the script above, on
one H100. Exact prompts, measurements and an ASR cross-check are in
[`examples/espnet/demo_assets/README.md`](examples/espnet/demo_assets/README.md).

| clip | what it says | length |
| --- | --- | --- |
| [`bagpiper_tts_hello_greeting.wav`](examples/espnet/demo_assets/bagpiper_tts_hello_greeting.wav) | "Hello, how are you today?" | 1.40 s |
| [`bagpiper_tts_train_announcement.wav`](examples/espnet/demo_assets/bagpiper_tts_train_announcement.wav) | "The next train to Boston departs from platform nine." | 2.78 s |
| [`bagpiper_tts_weather_report.wav`](examples/espnet/demo_assets/bagpiper_tts_weather_report.wav) | "Tomorrow will be cloudy with a high of eighteen degrees…" | 4.16 s |
| [`bagpiper_tts_numbers_and_date.wav`](examples/espnet/demo_assets/bagpiper_tts_numbers_and_date.wav) | "Your order total is one thousand two hundred thirty-four dollars…" | 8.14 s |
| [`bagpiper_tts_hello_greeting_cfg3.wav`](examples/espnet/demo_assets/bagpiper_tts_hello_greeting_cfg3.wav) | same as the first, with `--cfg 3.0` | 1.24 s |

GitHub does not play `.wav` inline; click a link to download, or clone and play
locally. Whisper transcribed the first two clips word-for-word (0.0% error).

## Docker

[`examples/espnet/docker/`](examples/espnet/docker/README.md) holds one
Dockerfile covering **x86_64/amd64 and ARM64/aarch64**, deriving from
`vllm/vllm-openai:v0.28.0` — verified against the Docker Hub registry API to be
a real multi-arch manifest list, with both per-arch digests pinned in
`build.sh`.

```bash
examples/espnet/docker/build.sh --arch amd64     # or arm64, or both
examples/espnet/docker/build.sh --arch arm64 --dry-run   # print the command only
```

**No image has been built or published from this repository, for either
architecture.** What is verified is the base image's architecture support, the
aarch64 availability of every pinned wheel, and the syntax of the Dockerfile and
build script. Runtime behaviour inside a container is not. The Docker README
splits verified from unverified explicitly.

## What has actually been tested

On H100 80GB (Linux, CUDA 13 driver, Python 3.12), tensor parallel size 1:

- **bagpiper conversion** — `espnet/bagpiper-tts-sft` and `espnet/bagpiper-sft`
  both converted from their official `model.pt`. Verified by reading the written
  safetensors back: 1381 of 1381 tensors bit-identical to the source, zero dtype
  or shape drift, `vocab_weight` dropped, shard index consistent.
- **bagpiper serving and audio** — the converted `bagpiper-tts-sft` served on this
  fork produced audio for 6 of 6 hand-written scene prompts, all
  `finish_reason=stop`. Whisper transcribed two of them word-for-word; the rest
  differ only in numeral spelling. Clips and numbers in `demo_assets/`.
- **`espnet/bagpiper-sft` audio does not work here** — 0 of 6 requests returned
  audio through the same server. See
  [`examples/espnet/MODELS.md`](examples/espnet/MODELS.md); use `bagpiper-tts-sft`
  for speech.
- **opuslm**, **opuslm_dialogue** — provenance proven by hash and tensor
  comparison (see MODELS.md). Their earlier TTS/ASR/dialogue runs are recorded in
  the Chinese guide; they were **not** re-run for this change.
- Docker — **not** built end-to-end; static and config checks only.
- Not covered: multi-GPU (TP>1), throughput or latency benchmarking, and formal
  audio-quality scoring. Nobody listened to the demo clips as part of producing
  them; the checks are measurements plus the Whisper cross-check.

## More

- [`examples/espnet/GETTING_STARTED.zh.md`](examples/espnet/GETTING_STARTED.zh.md)
  — the full guide (Chinese): what the models are, conversion details, runnable
  examples with measured output, Docker on a personal machine, verification
  status, and known limitations.
- [`examples/espnet/MODELS.md`](examples/espnet/MODELS.md) — which official repo
  and revision each model comes from, proven by hash and tensor comparison, plus
  what conversion each needs and the one known gap.
- [`examples/espnet/demo_assets/README.md`](examples/espnet/demo_assets/README.md)
  — the demo clips: exact prompts, measurements, ASR cross-check.
- [`examples/espnet/README.md`](examples/espnet/README.md) — tooling layout.
- [`README.vllm.md`](README.vllm.md) — upstream vLLM's README, verbatim.

Licensed under Apache-2.0, the same as upstream vLLM. See [LICENSE](LICENSE).
