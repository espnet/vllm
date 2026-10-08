# OWSM autoregressive inference

OWSM uses ESPnet's waveform frontend, normalization and encoder with a vLLM
Transformer decoder. Self-attention and cross-attention use vLLM KV caches.
Decoder biases, sinusoidal positions, embedding scaling and ESPnet's LayerNorm
epsilon are preserved. The initial backend requires TP=PP=1 and eager execution.

The inference example defaults to float32/TRITON_ATTN, the validated backend.
`--dtype` and `--attention-backend` select other numerical paths explicitly.
Float32 inference explicitly uses `TRITON_F32_DEFAULT=ieee` and disables
torch/cuBLAS TF32. Torch flags alone do not control Triton's `tl.dot` default.
On a fixed-action v4 diagnosis with the same real audio and 173 retained actions,
the maximum native/vLLM logprob difference decreased from 0.03504658 with
Triton TF32 to 0.00010395 with IEEE. This is a numerical diagnosis; each RL run
still requires its own probability gate and update/export validation.
Direct Python API callers should set these precision controls before engine
construction when matching the native float32 policy.

The same adapter constructs the v4 base/small/medium, v3.2, v3.1 E-Branchformer,
v3 Transformer, v2 Transformer/E-Branchformer and v1 Transformer configurations.
Nine public checkpoints pass strict native weight loading and real H100 engine
checks with float32, TRITON_ATTN and fixed training-window inputs: v1, v2,
v2 E-Branchformer, v3, v3.1 E-Branchformer small, v3.2, and v4 base/small/medium.
The v4 base and v3.1 small checkpoints also pass variable-length checks and have
CPU decoder numerical evidence. Each engine run compares 11 cases, including
exact greedy tokens, actual sampled-token raw logprobs, reversed input order,
and delayed request insertion, with a fixed logprob error limit of 0.03 for
32-token generation. Delayed insertion does not prove simultaneous encoder
prefill and old-request decode in the same scheduler batch; production ordering
also has dedicated regression tests. Dynamic inputs on the other seven
checkpoints have not been evaluated.
BF16/FlashAttention has not passed strict native token/logprob equivalence:
reduced-precision attention and incremental execution can change rounding and
greedy choices. OWSM-CTC is a separate architecture and is rejected.

Install a checkpoint-compatible ESPnet release in the inference environment.
Use `uv` and the vLLM checkout's documented precompiled Python installation.
The native frontend uses float32 even when encoder/decoder weights use bf16.
Sinusoidal caches are initialized on CPU in float32, matching native ESPnet
initialization before the model moves to its inference device and precision.

Download a native checkpoint, saved training config, SentencePiece model and
normalization statistics. Set the config's `bpemodel` and `normalize_conf.stats_file`
to downloaded paths before exporting. The export preserves native tensor names
and embeds the complete model configuration and token list. Inference uses
checkpoint normalization buffers without consulting external statistics paths.

```bash
PYTHONPATH="$PWD" .venv/bin/python examples/espnet/export_owsm.py \
  --config /path/to/owsm/config.yaml \
  --checkpoint /path/to/owsm/valid.total_count.ave_5best.pth \
  --output /path/to/owsm-vllm --model-version v4

PYTHONPATH="$PWD" .venv/bin/python examples/espnet/infer_owsm.py \
  --model /path/to/owsm-vllm --audio /path/to/mono.wav --language eng
```

The default processor pads clips to the checkpoint's training window, matching
ESPnet `Speech2Text._buffer`. It rejects oversized clips; split long recordings
before generation. With 30 seconds at 16 kHz, v4 reserves 374 encoder positions,
v3/v2 reserve 749, and v1 reserves 1498. Setting `pad_to_training_window=false`
in the exported config enables variable-length clips and changes conditioning.

The decoder prompt is explicit: `<sos><eng><asr><notimestamps>` for v3+ and
`<sos><en><asr><notimestamps>` for v1/v2. The example resolves `en`, `eng` or
`english` to the symbol in the checkpoint vocabulary; translation tasks such as
`st_en`/`st_eng` follow the same rule. Exact symbols take precedence, and missing
or ambiguous aliases raise an error. For Python APIs, pass an
`ExplicitEncoderDecoderPrompt` with an encoder placeholder `[0]` and audio, plus
decoder token IDs including `<sos>`. No extra start token is injected.

This backend samples the autoregressive decoder. It does not implement ESPnet's
hybrid CTC/attention beam search, external language-model fusion, timestamp
pairing filters or streaming segmentation. Compare AR logits and generation
under identical sampling constraints when assessing equivalence. The example
suppresses timestamp tokens for no-timestamps inference. RL must also recompute
the probabilities of its declared sampling distribution.

CPU tests:

```bash
PYTHONPATH="$PWD:/path/to/espnet" .venv/bin/python -m pytest \
  tests/models/multimodal/test_owsm_decoder.py \
  tests/models/multimodal/test_owsm_processing.py -q
```

They verify decoder numerical parity, incremental reference caches, token-ID
fidelity, fixed-window processing and encoder placeholder lengths. Production
GPU paged attention requires a separate real-checkpoint generation test.
