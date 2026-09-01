# ESPnet audio-LM docker image

Derives from `vllm/vllm-openai:v0.28.0`, swaps in this fork's Python code
(native extensions are reused from the official v0.28.0 wheel — the fork
is Python-only), and adds the ESPnet runtime dependencies for OpusLM
audio decode/encode.

## Build

From the **repo root** (the build context must be the repo root):

```bash
docker build -f examples/espnet/docker/Dockerfile -t espnet-vllm:v0.28.0 .
```

This `COPY`s the local checkout, so the image contains exactly your
working tree. To build from a remote instead, edit the Dockerfile: replace
the `COPY . /workspace/vllm-fork` line with the `ARG` + `git clone` block
in the comment above it, then pass `--build-arg VLLM_FORK_URL=...
--build-arg VLLM_FORK_REF=...`. The build args have no effect on their
own — nothing reads them until that line is replaced.

## Run

The image keeps the upstream `ENTRYPOINT ["vllm", "serve"]`, so arguments
after the image name go straight to `vllm serve`:

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/models \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 9812:9812 \
    espnet-vllm:v0.28.0 \
    /models/OpusLM \
    --served-model-name opuslm \
    --trust-remote-code \
    --max-model-len 8192 \
    --port 9812 \
    --no-async-scheduling \
    --limit-mm-per-prompt '{"audio": 8}' \
    --enable-prefix-caching
```

Notes:

- `--no-async-scheduling` matches the default of the `serve_*.sh` scripts
  (`ASYNC_SCHEDULING=0`). Async scheduling is on by default in v0.28.0;
  this flag selects the configuration the audio path was brought up on.
- `VLLM_USE_V2_MODEL_RUNNER=0` is baked into the image with `ENV`, because
  the audio hooks only exist in the V1 model runner and a plain `docker run`
  bypasses the serve scripts that would otherwise set it. Do not override it
  for these three models: the V2 runner boots and emits text but returns no
  audio at all.

- The HF cache mount matters: bagpiper lazy-loads the Xcodec decoder from
  `hf-audio/xcodec-hubert-general` on the first audio decode, and OpusLM
  audio *input* pulls the XEUS checkpoint + kmeans model from `espnet/xeus`
  (2.2 GB). Without a warm cache the container needs network access to the
  HF hub for those.
- OpusLM's DAC decoder is the exception, and it is already baked into the
  image. It is fetched through `espnet_model_zoo`, which keeps its own
  cache inside `site-packages/espnet_model_zoo` rather than the HF hub
  cache, so the `~/.cache/huggingface` mount above would not cover it. The
  build pre-warms it (308 MB), so audio output works with no egress.
- Adjust `-p` and `--port` per model (defaults used by the serve scripts
  and clients: bagpiper 9811, opuslm 9812, opuslm_dialogue 9813).
- H100 (sm90) is covered by the upstream v0.28.0 wheels; no local
  compilation happens in this image.
- To use the serve scripts inside the container, override the entrypoint:
  `docker run --entrypoint bash ... -c 'MODEL_PATH=/models/OpusLM bash /workspace/vllm-fork/examples/espnet/serve_opuslm.sh'`.
