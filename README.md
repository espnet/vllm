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

| name | backbone | what it does |
| --- | --- | --- |
| `bagpiper` | Qwen3-8B + Qwen3-Omni audio tower + Xcodec | text and audio generation from a described scene; 8 codec streams, optional CFG |
| `opuslm` | OLMo-2-7B | TTS, ASR, text LM |
| `opuslm_dialogue` | SmolLM2-1.7B | spoken dialogue |

Use those exact names for `--served-model-name`, and make sure `config.json`
declares the matching `model_type`.

## Quickstart

```bash
# 1. ESPnet/DeepSpeed checkpoint -> HF-style directory
python examples/espnet/convert/convert_bagpiper_ckpt.py \
    /path/to/mp_rank_00_model_states.pt /path/to/out/bagpiper \
    --ref-dir /path/to/reference/bagpiper-checkpoint

# 2. serve (port 9811; extra args pass through to `vllm serve`)
MODEL_PATH=/path/to/out/bagpiper bash examples/espnet/serve_bagpiper.sh

# 3. request
python examples/espnet/clients/client_bagpiper.py --task tts --out demo.wav
```

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

Output of the built-in default prompt (`'Hello, how are you today?'`, 1.80 s,
16 kHz mono):
[`examples/espnet/demo_assets/bagpiper_hello_scene.wav`](examples/espnet/demo_assets/bagpiper_hello_scene.wav)

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

- **bagpiper** — validated 2026-09-02 against the release checkpoint with the
  authoritative prompt format: 5 of 5 audio-generation requests returned audio,
  all `finish_reason=stop`, implied speaking rates 2.78 / 3.06 / 2.84 words per
  second where the spoken span was measurable.
- **opuslm**, **opuslm_dialogue** — TTS, ASR, text-LM and spoken-dialogue paths
  exercised with token counts and durations recorded in the guide.
- Docker — **not** built end-to-end; static and config checks only.
- Not covered: multi-GPU (TP>1), throughput or latency benchmarking, and any
  formal audio-quality scoring. The recorded checks are measurable properties
  (duration, sample rate, channels, finish reason, speaking-rate arithmetic),
  not listening tests.

## More

- [`examples/espnet/GETTING_STARTED.zh.md`](examples/espnet/GETTING_STARTED.zh.md)
  — the full guide (Chinese): what the models are, conversion details, runnable
  examples with measured output, Docker on a personal machine, verification
  status, and known limitations.
- [`examples/espnet/README.md`](examples/espnet/README.md) — tooling layout.
- [`README.vllm.md`](README.vllm.md) — upstream vLLM's README, verbatim.

Licensed under Apache-2.0, the same as upstream vLLM. See [LICENSE](LICENSE).
