# Container validation — 2026-09-10

Validated source: [`09efdff51f`](https://github.com/espnet/vllm/commit/09efdff51f5d3f6dceeaa119323c1f92eb05fc1e).
The runtime is vLLM 0.28.0, torch 2.13.0+cu130, CUDA 13.0 and the explicitly
versioned ESPnet serving package `202609.post1+vllm.0.28.0` described in
[COMPATIBILITY.md](COMPATIBILITY.md).

## Native builds

Both architecture jobs in [GitHub Actions run 34443516445](https://github.com/espnet/vllm/actions/runs/34443516445)
succeeded on standard `ubuntu-24.04` and `ubuntu-24.04-arm` runners. Each job
built the full image and repeated dependency, model/codec/SSL import and CLI
checks in containers with networking disabled. Publication was skipped on the
test branch.

The GPU tests use native images built from the same Dockerfile and source,
with the checkout supplied as a checksum-verified GitHub archive instead of
local `COPY`. The archive SHA256 is
`62d127081c1166e5c6dadbfad30024266fc670cec965e15ebcd384f4460af087`.
The images contain the complete installed runtime; GPU tests do not install
or replace Python dependencies.

| Architecture | GPU image manifest digest |
| --- | --- |
| amd64 | `sha256:116caf06104ca726d478bb4c6cc99e7cbb7e8d422b71dbdff7edf57ad10ed3e2` |
| arm64 | `sha256:f04ca0d6b4a685771bf2d4699d855b16e8aa11f58d0a156e7309df82c10a69c2` |

## GPU requests

Checkpoints are the converted `espnet/bagpiper-tts-sft`,
`espnet/OpusLM_7B_Anneal` and `espnet/multi_turn_SDS_RLAIF` models documented
in [MODELS.md](../MODELS.md). Requests use the repository's reference clients.
The server configuration is:

```bash
vllm serve /models/MODEL --served-model-name MODEL_NAME \
    --trust-remote-code --max-model-len 8192 --max-num-seqs 4 \
    --no-async-scheduling --limit-mm-per-prompt '{"audio":1}' \
    --enforce-eager --gpu-memory-utilization 0.65 --seed 0
```

The image supplies `VLLM_USE_V2_MODEL_RUNNER=0`. Each model runs alone on one
GPU. Request payloads also set `seed=0`. Bagpiper uses the reference client's
default greeting prompt and system message, audio temperature 0.8, top-k 20,
CFG 1 and a 4096-token limit. OpusLM uses “Hello world” for TTS. ASR and audio
dialogue use `demo_assets/bagpiper_tts_hello_greeting.wav`; text dialogue uses
“How are you today?”. Other requests use audio temperature 0.8, top-k 30
and a 2048-token limit.

| Request | H100 / amd64 | GB200 / arm64 |
| --- | --- | --- |
| Bagpiper TTS | 1.22 s WAV | 1.90 s WAV |
| OpusLM TTS | 6.20 s WAV | 6.56 s WAV |
| OpusLM ASR | “Hello, how are you today?” | “Hello, how are you today?” |
| Dialogue: text | Nonempty text, no audio | Nonempty text, no audio |
| Dialogue: audio | 0.90 s WAV | 2.08 s WAV |

All ten requests returned HTTP 200 and `finish_reason=stop`. Generated audio
passed RIFF/WAV parsing and checks for nonzero duration and nonsilent samples.
Text dialogue is a text-output task; absence of audio in that mode is expected.
All generated WAVs use 16 kHz PCM audio.

The H100 Bagpiper WAV is byte-identical to the previously validated H100
normal-generation result, SHA256
`86428f178aab29a4bbd28fbfa76c02432fa7129697551d241b68756415c332d2`.
This comparison covers one fixed prompt and seed. Cross-GPU outputs need not
be byte-identical, and these short functional checks do not establish general
speech quality, concurrency, CFG, preemption or throughput performance.

## Public image status

These build and GPU checks are separate from Docker Hub publication. The
first public `espnet/vllm:latest` requires the owner's two setup steps in
[OWNER_SETUP.md](OWNER_SETUP.md), followed by a successful publishing run and
verification that the public manifest lists both architectures.
