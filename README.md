# vLLM + ESPnet audio language models

A fork of [vLLM](https://github.com/vllm-project/vllm) that adds serving support
for three ESPnet speech models: **bagpiper**, **opuslm**, and
**opuslm_dialogue**, with engine and serving extensions for multi-stream audio
generation and classifier-free guidance (CFG).

This repository accompanies [An Efficient vLLM-Based Inference Pipeline for
Unified Audio Understanding and Generation](https://arxiv.org/abs/2607.02119),
accepted at **Interspeech 2026**. See [Citation](#citation).

> This is a modified fork, not an official vLLM release. Upstream's own README is
> preserved verbatim at [README.vllm.md](README.vllm.md).

## Versions

| Component | Version |
| --- | --- |
| Upstream base | vLLM **v0.28.0**, tag commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` (2026-08-23) |
| Default branch | `main` |
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
step is required. The converter builds those missing files itself, from public
sources pinned by revision and hash, so nothing else is needed.
[`examples/espnet/MODELS.md`](examples/espnet/MODELS.md) records which repo and
revision each model comes from, how that was proven, and where every generated
file comes from.

## Quickstart

Copy-paste runnable on a clean machine once vLLM 0.28.0 and this fork's Python
tree are installed. One GPU, no other inputs.

```bash
# 1. fetch the official weights (17 GB of native ESPnet checkpoint)
hf download espnet/bagpiper-tts-sft --local-dir ~/models/bagpiper-tts-sft
(cd ~/models/bagpiper-tts-sft && sha256sum -c SHA256SUMS)

# 2. convert. config.json and the tokenizer are built from pinned public
#    sources -- Qwen3-8B-Base for the text side, Qwen3-Omni for the audio
#    tower, both named by Bagpiper's own released training YAML -- then
#    validated before any weight is written.
python examples/espnet/convert/convert_bagpiper_ckpt.py \
    ~/models/bagpiper-tts-sft ~/models/bagpiper-tts-sft-vllm
# add --dry-run first to check weight coverage without writing 17 GB

# 3. serve (port 9811; extra args pass through to `vllm serve`)
MODEL_PATH=~/models/bagpiper-tts-sft-vllm bash examples/espnet/serve_bagpiper.sh

# 4. request
python examples/espnet/clients/client_bagpiper.py --task tts --out demo.wav
```

The other two models are the same shape, with `--model` naming which one the
checkpoint is:

```bash
hf download espnet/OpusLM_7B_Anneal --local-dir ~/models/opuslm
python examples/espnet/convert/convert_opuslm_ckpt.py \
    ~/models/opuslm ~/models/opuslm-vllm --model opuslm
MODEL_PATH=~/models/opuslm-vllm bash examples/espnet/serve_opuslm.sh

hf download espnet/multi_turn_SDS_RLAIF --local-dir ~/models/sds
python examples/espnet/convert/convert_opuslm_ckpt.py \
    ~/models/sds ~/models/opuslm-dialogue-vllm --model opuslm_dialogue
MODEL_PATH=~/models/opuslm-dialogue-vllm bash examples/espnet/serve_opuslm_dialogue.sh
```

Two things the converter refuses to do quietly. It validates every tensor
against the weight groups the model can load and exits non-zero listing
anything unexpected, so a layout change fails loudly instead of producing a
directory that loads with missing weights. And every fetched source file is
checked against a recorded sha256, so an upstream edit to a tokenizer stops the
conversion instead of silently changing your model.

On a machine with no network access, pre-download what
`python examples/espnet/convert/bootstrap_assets.py --model bagpiper --print-sources`
lists (about 11 MB) and pass `--assets-from <dir>`. To build just the
config/tokenizer, or to diff them against a directory you already trust, run
that script directly.

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

The repository records static checks of the base image, dependencies and build
scripts. A locally built image has been reported by the maintainer; its tag,
source revision and runtime results still need to be recorded here.
See [Docker Hub automation](examples/espnet/docker/PUBLISHING.md) for the
build, validation and publication workflow and its setup requirements.

## What has actually been tested

On H100 80GB (Linux, CUDA 13 driver, Python 3.12), tensor parallel size 1:

- **bagpiper conversion** — `espnet/bagpiper-tts-sft` and `espnet/bagpiper-sft`
  both converted from their official `model.pt`. Verified by reading the written
  safetensors back: 1381 of 1381 tensors bit-identical to the source, zero dtype
  or shape drift, `vocab_weight` dropped, shard index consistent.
- **conversion with no reference directory** — `espnet/bagpiper-tts-sft`
  converted again from `model.pt` alone, with `config.json` and the tokenizer
  fetched and built from the pinned public sources. 1381 of 1381 tensors
  bit-identical to the source, and all four safetensors shards sha256-identical
  to the earlier conversion. The generated assets were diffed against the
  known-good directory file by file (see MODELS.md) and the loaded tokenizers
  agree on every id tested. Run on transformers 5.16.1 / huggingface_hub
  1.29.0, i.e. not the version the assets were originally written with.
- **serving and audio from that directory** — served on this fork and returned
  audio for both requests sent, `finish_reason=stop`: 1.26 s for the client's
  default scene prompt and 2.58 s for a fresh hand-written one. Whisper
  transcribed the first word-for-word; on the second it dropped a leading "The"
  and wrote "6" for "six". Nobody listened to these two clips.
- **bagpiper serving and audio** — the converted `bagpiper-tts-sft` served on this
  fork produced audio for 6 of 6 hand-written scene prompts, all
  `finish_reason=stop`. Whisper transcribed two of them word-for-word; the rest
  differ only in numeral spelling. Clips and numbers in `demo_assets/`.
- **`espnet/bagpiper-sft` audio does not work here** — 0 of 6 requests returned
  audio through the same server. See
  [`examples/espnet/MODELS.md`](examples/espnet/MODELS.md); use `bagpiper-tts-sft`
  for speech.
- **opuslm_dialogue conversion** — converted from the official `2epoch.pth`
  with no reference directory: 220 tensors in 2 shards, and the generated
  `config.json` came out **identical** to the known-good one, key for key.
- **opuslm**, **opuslm_dialogue** — provenance proven by hash and tensor
  comparison (see MODELS.md). Their generated assets were diffed against the
  known-good directories, but neither model was **served** for this change; the
  earlier TTS/ASR/dialogue runs are recorded in the Chinese guide.
- Docker — recorded checks cover static configuration; validation of the
  maintainer's locally built image is pending.
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

## Citation

If you use this inference pipeline, please cite our Interspeech 2026 paper:

```bibtex
@inproceedings{wang2026efficient,
  title={An Efficient {vLLM}-Based Inference Pipeline for Unified Audio Understanding and Generation},
  author={Wang, Haoran and Tian, Jinchuan and Arora, Siddhant and Watanabe, Shinji},
  booktitle={Interspeech 2026},
  year={2026},
  note={Accepted for publication},
  eprint={2607.02119},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2607.02119}
}
```
