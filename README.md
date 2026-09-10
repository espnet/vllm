# vLLM + ESPnet

A [vLLM](https://github.com/vllm-project/vllm) integration for ESPnet audio
language models, supporting speech synthesis, speech recognition, and spoken
dialogue through an OpenAI-compatible API. This fork extends the inference
engine with multi-stream audio generation and classifier-free guidance (CFG).

This repository accompanies [An Efficient vLLM-Based Inference Pipeline for
Unified Audio Understanding and Generation](https://arxiv.org/abs/2607.02119),
accepted at **Interspeech 2026**.

## Supported models

| Model | Backbone | Tasks | Checkpoint |
| --- | --- | --- | --- |
| `bagpiper` | Qwen3-8B + Qwen3-Omni audio tower | Scene-conditioned speech generation with optional CFG; text and audio understanding | [`espnet/bagpiper-tts-sft`](https://huggingface.co/espnet/bagpiper-tts-sft) |
| `opuslm` | OLMo-2-7B | Text-to-speech, speech recognition, text continuation | [`espnet/OpusLM_7B_Anneal`](https://huggingface.co/espnet/OpusLM_7B_Anneal) |
| `opuslm_dialogue` | SmolLM2-1.7B | Spoken and text dialogue | [`espnet/multi_turn_SDS_RLAIF`](https://huggingface.co/espnet/multi_turn_SDS_RLAIF) |

Use the model names above for `--served-model-name` and the matching
`model_type` in `config.json`. Official ESPnet checkpoints require conversion
before serving; the converters generate the configuration, tokenizer assets,
and safetensors files. See [Model checkpoints and conversion](examples/espnet/MODELS.md)
for source revisions and conversion details.

## Installation

This integration is based on **vLLM 0.28.0** and uses its precompiled CUDA
extensions. The Docker image packages the fork with the ESPnet serving
dependencies for **Linux amd64 and arm64**.

Build from the repository root, selecting the host architecture:

```bash
examples/espnet/docker/build.sh --arch amd64 --tag espnet-vllm:v0.28.0
# On ARM64:
examples/espnet/docker/build.sh --arch arm64 --tag espnet-vllm:v0.28.0
```

See the [Docker guide](examples/espnet/docker/README.md) for GPU requirements,
container launch commands, and model-cache configuration. The image includes
the runtime and codecs; download and convert the language-model checkpoints
separately.

## Quickstart

Run the following commands from the repository root in an environment with
this fork and its ESPnet dependencies installed. In the Docker image, the
repository is located at `/workspace/vllm-fork`.

### Bagpiper

```bash
# Download and verify the checkpoint.
hf download espnet/bagpiper-tts-sft --local-dir ~/models/bagpiper-tts-sft
(cd ~/models/bagpiper-tts-sft && sha256sum -c SHA256SUMS)

# Convert the checkpoint and generate the required configuration and tokenizer.
python examples/espnet/convert/convert_bagpiper_ckpt.py \
    ~/models/bagpiper-tts-sft ~/models/bagpiper-tts-sft-vllm

# Start the server on port 9811.
MODEL_PATH=~/models/bagpiper-tts-sft-vllm bash examples/espnet/serve_bagpiper.sh
```

Send a request from a second terminal:

```bash
python examples/espnet/clients/client_bagpiper.py --task tts --out demo.wav
```

Bagpiper generates audio from a scene description. To specify spoken content,
include the line in quotation marks within the description:

```bash
python examples/espnet/clients/client_bagpiper.py --task tts --out demo.wav \
    --prompt "A calm male voice, close-miked in a quiet studio, says: \
'Your package will arrive on Tuesday.' No background noise."
```

The client supplies the audio-generation system prompt by default. Use
`--task tts_cfg --cfg 3.0` to enable CFG. Speech generation uses
`espnet/bagpiper-tts-sft`; audio output from `espnet/bagpiper-sft` is currently
unsupported by this integration.

### OpusLM

```bash
hf download espnet/OpusLM_7B_Anneal --local-dir ~/models/opuslm
python examples/espnet/convert/convert_opuslm_ckpt.py \
    ~/models/opuslm ~/models/opuslm-vllm --model opuslm
MODEL_PATH=~/models/opuslm-vllm bash examples/espnet/serve_opuslm.sh
```

### OpusLM-dialogue

```bash
hf download espnet/multi_turn_SDS_RLAIF --local-dir ~/models/sds
python examples/espnet/convert/convert_opuslm_ckpt.py \
    ~/models/sds ~/models/opuslm-dialogue-vllm --model opuslm_dialogue
MODEL_PATH=~/models/opuslm-dialogue-vllm bash examples/espnet/serve_opuslm_dialogue.sh
```

The converters verify source-asset checksums and checkpoint tensor coverage.
Use `--dry-run` to inspect conversion without writing weights. For offline
conversion, use `bootstrap_assets.py --model <name> --print-sources` to list
required assets, download them in advance, and pass `--assets-from <dir>`.
Additional request examples are in the [client documentation](examples/espnet/clients/README.md).

## Serving configuration

The launch scripts select the V1 model runner
(`VLLM_USE_V2_MODEL_RUNNER=0`) and synchronous scheduling
(`--no-async-scheduling`). These audio models require the V1 runner. Audio is
returned as a complete WAV in non-streaming responses. Text-dialogue requests
return text without audio.

Container compatibility and GPU smoke-test coverage are documented in
[Runtime compatibility](examples/espnet/docker/COMPATIBILITY.md) and
[Container verification](examples/espnet/docker/VALIDATION.md). Performance
and audio-quality benchmarks are outside the scope of those functional checks.

## Audio examples

| Example | Duration |
| --- | --- |
| [Greeting](examples/espnet/demo_assets/bagpiper_tts_hello_greeting.wav) | 1.40 s |
| [Greeting with CFG](examples/espnet/demo_assets/bagpiper_tts_hello_greeting_cfg3.wav) | 1.24 s |
| [Train announcement](examples/espnet/demo_assets/bagpiper_tts_train_announcement.wav) | 2.78 s |
| [Weather report](examples/espnet/demo_assets/bagpiper_tts_weather_report.wav) | 4.16 s |
| [Order total and date](examples/espnet/demo_assets/bagpiper_tts_numbers_and_date.wav) | 8.14 s |

Generated with `espnet/bagpiper-tts-sft` on an H100 GPU. Download the WAV files
to listen; prompts and generation settings are in the
[audio examples documentation](examples/espnet/demo_assets/README.md).

## Documentation

- [Getting started (中文)](examples/espnet/GETTING_STARTED.zh.md)
- [Model checkpoints and conversion](examples/espnet/MODELS.md)
- [Reference clients](examples/espnet/clients/README.md)
- [Docker build and deployment](examples/espnet/docker/README.md)
- [Docker Hub publishing](examples/espnet/docker/PUBLISHING.md)
- [Upstream vLLM documentation](README.vllm.md)

## Citation

If you use this work, please cite:

```bibtex
@article{wang2026efficient,
  title={An Efficient vLLM-Based Inference Pipeline for Unified Audio Understanding and Generation},
  author={Wang, Haoran and Tian, Jinchuan and Arora, Siddhant and Watanabe, Shinji},
  journal={arXiv preprint arXiv:2607.02119},
  year={2026}
}
```

## License

[Apache License 2.0](LICENSE). This repository is maintained as an ESPnet fork
of vLLM; the upstream README is preserved in [README.vllm.md](README.vllm.md).
