# Container verification

Container checks cover dependency consistency, runtime imports, and functional
inference. GPU checks are recorded for a specific source revision and runtime;
they do not constitute performance or audio-quality benchmarks.

## Reference configuration

| Component | Value |
| --- | --- |
| Verification date | 2026-09-10 |
| Source revision | [`09efdff51f`](https://github.com/espnet/vllm/commit/09efdff51f5d3f6dceeaa119323c1f92eb05fc1e) |
| vLLM | 0.28.0 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0 |
| ESPnet serving package | 202609.post1+vllm.0.28.0 |
| Platforms | H100 / amd64; GB200 / arm64 |

Package constraints are documented in [COMPATIBILITY.md](COMPATIBILITY.md).
GPU tests used native images built from the same Dockerfile and source,
without installing or replacing runtime dependencies during testing.

## Build checks

[GitHub Actions run 34443516445](https://github.com/espnet/vllm/actions/runs/34443516445)
completed native builds on `ubuntu-24.04` and `ubuntu-24.04-arm`. Each image
passed dependency checks, model and codec imports, and CLI initialization.
These checks were repeated in containers with networking disabled.

## GPU smoke tests

Checkpoints were converted from `espnet/bagpiper-tts-sft`,
`espnet/OpusLM_7B_Anneal`, and `espnet/multi_turn_SDS_RLAIF` as documented in
[MODELS.md](../MODELS.md). Each model ran on one GPU with this configuration:

```bash
vllm serve /models/MODEL --served-model-name MODEL_NAME \
    --trust-remote-code --max-model-len 8192 --max-num-seqs 4 \
    --no-async-scheduling --limit-mm-per-prompt '{"audio":1}' \
    --enforce-eager --gpu-memory-utilization 0.65 --seed 0
```

The image sets `VLLM_USE_V2_MODEL_RUNNER=0`. Requests used the reference
clients' payloads with `seed=0`:

- Bagpiper: default greeting and system prompt, audio temperature 0.8,
  top-k 20, CFG 1, and a 4096-token limit.
- OpusLM TTS: “Hello world”.
- ASR and audio dialogue: `demo_assets/bagpiper_tts_hello_greeting.wav`.
- Text dialogue: “How are you today?”.

OpusLM and dialogue requests used audio temperature 0.8, top-k 30, and a
2048-token limit.

| Request | H100 / amd64 | GB200 / arm64 |
| --- | --- | --- |
| Bagpiper speech generation | Pass | Pass |
| OpusLM TTS | Pass | Pass |
| OpusLM ASR | Pass | Pass |
| Text dialogue | Pass | Pass |
| Audio dialogue | Pass | Pass |

All requests returned HTTP 200 and `finish_reason=stop`. Audio-output checks
required a valid 16 kHz PCM WAV with nonzero duration and nonsilent samples.
ASR returned “Hello, how are you today?” on both platforms. Text dialogue
returned nonempty text without audio, as required by that task.

## Coverage

The GPU suite covers one request per task and platform. It does not evaluate
multi-GPU serving, concurrent requests, CFG, preemption, throughput, or general
speech quality. Re-run GPU checks when changing model code, the ESPnet serving
package, or the CUDA runtime. Docker Hub publication is verified separately
using the commands in [PUBLISHING.md](PUBLISHING.md).
