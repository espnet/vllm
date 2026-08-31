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

Alternative: instead of `COPY`ing the local checkout, the Dockerfile
documents a `git clone` variant driven by `VLLM_FORK_URL` /
`VLLM_FORK_REF` build args — see the comments in the Dockerfile.

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
    --limit-mm-per-prompt '{"audio": 8}' \
    --enable-prefix-caching
```

Notes:

- The HF cache mount matters: bagpiper lazy-loads the Xcodec decoder from
  `hf-audio/xcodec-hubert-general` on the first audio decode; OpusLM
  lazy-loads the ESPnet DAC codec, and audio input additionally pulls the
  XEUS checkpoint + kmeans model from `espnet/xeus`. Without a warm cache
  the container needs network access to the HF hub.
- Adjust `-p` and `--port` per model (defaults used by the serve scripts
  and clients: bagpiper 9811, opuslm 9812, opuslm_dialogue 9813).
- H100 (sm90) is covered by the upstream v0.28.0 wheels; no local
  compilation happens in this image.
- To use the serve scripts inside the container, override the entrypoint:
  `docker run --entrypoint bash ... -c 'MODEL_PATH=/models/OpusLM bash /workspace/vllm-fork/examples/espnet/serve_opuslm.sh'`.
