# OWSM autoregressive inference

OWSM uses ESPnet's waveform frontend, normalization and encoder with a vLLM
Transformer decoder. Self-attention and cross-attention use vLLM KV caches.
Decoder biases, sinusoidal positions, embedding scaling and ESPnet's LayerNorm
epsilon are preserved. The initial backend requires TP=PP=1 and eager execution.

The same adapter constructs the v4 base/small/medium, v3.2, v3.1 E-Branchformer,
v3 Transformer, v2 Transformer/E-Branchformer and v1 Transformer configurations.
Real pretrained weight and numerical validation covers v4 base 102M and
v3.1 E-Branchformer small. Their complete native state loads strictly, including
the older subsampling-key layout; decoder logits match ESPnet on CPU. The other
configurations have constructor checks. GPU engine validation is pending. OWSM-CTC is a separate architecture and is rejected.

Install a checkpoint-compatible ESPnet release in the inference environment.
Use `uv` and the vLLM checkout's documented precompiled Python installation.
The native frontend uses float32 even when encoder/decoder weights use bf16.

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

The decoder prompt is explicit: `<sos><eng><asr><notimestamps>`. Other tasks use
their checkpoint's language and task symbols. For Python APIs, pass an
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
