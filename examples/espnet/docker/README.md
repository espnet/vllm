# ESPnet audio-LM docker image

Derives from `vllm/vllm-openai:v0.28.0`, swaps in this fork's Python code
(native extensions are reused from the official v0.28.0 wheel — the fork
is Python-only), and adds the ESPnet runtime dependencies for OpusLM
audio decode/encode.

Supports **x86_64/amd64** and **ARM64/aarch64** from a single Dockerfile.

## Architectures

The base image is a genuine multi-arch manifest list. Verified against the
Docker Hub registry API on 2026-09-03 — not assumed:

| | |
| --- | --- |
| `mediaType` | `application/vnd.docker.distribution.manifest.list.v2+json` |
| index digest | `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` |
| `linux/amd64` | `sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635` |
| `linux/arm64` | `sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce` |

Each per-arch **image config blob** was fetched too, and reports
`architecture: amd64` / `arm64` with `os: linux`. So the index is not merely
claiming two platforms — both are real.

That is why there is one Dockerfile rather than two. The only
per-architecture input is which base digest to pin, and `build.sh` supplies it
explicitly. Two near-identical Dockerfiles differing in one `FROM` line would
drift apart; this keeps the architectures visible without the duplication.

## Build

`build.sh` pins the base by digest, so a rebuild months from now uses the same
bytes even if the `v0.28.0` tag is re-pushed:

```bash
examples/espnet/docker/build.sh --arch amd64     # x86_64, digest-pinned, --load
examples/espnet/docker/build.sh --arch arm64     # aarch64, digest-pinned, --load
examples/espnet/docker/build.sh --arch both      # multi-platform, OCI layout
examples/espnet/docker/build.sh --arch amd64 --dry-run   # print the command only
```

A cross-architecture build needs QEMU binfmt registered
(`docker run --privileged --rm tonistiigi/binfmt --install all`), and
`--arch both` needs a buildx container driver (`docker buildx create --use`).
Multi-platform results cannot be `--load`ed into the classic image store, so
`--arch both` writes an OCI layout tarball instead of pushing anywhere.

The plain single-arch equivalent, run from the **repo root** (the build context
must be the repo root), resolves whichever architecture the daemon is on:

```bash
docker build -f examples/espnet/docker/Dockerfile -t espnet-vllm:v0.28.0 .
```

Either path `COPY`s the local checkout, so the image contains exactly your
working tree. To build from a remote instead, edit the Dockerfile: replace
the `COPY . /workspace/vllm-fork` line with the `ARG` + `git clone` block
in the comment above it, then pass `--build-arg VLLM_FORK_URL=...
--build-arg VLLM_FORK_REF=...`. The build args have no effect on their
own — nothing reads them until that line is replaced.

## What is verified, and what is not

**Verified** (static and registry/index checks, 2026-09-03):

- the base image carries both `linux/amd64` and `linux/arm64`, confirmed from
  the per-arch config blobs, with the digests above;
- `torch==2.13.0`, `torchvision==0.28.0` and `torchaudio==2.11.0` — the pins
  this image force-reinstalls — all publish `manylinux_2_28_aarch64` wheels as
  well as `x86_64`, so that layer is not x86-only by availability;
- `espnet` and `espnet_model_zoo` ship pure-Python `py3-none-any` wheels;
- `Dockerfile` parses: first instruction is `FROM`, no unknown instructions, no
  dangling line continuation;
- `build.sh` passes `bash -n`, and its `--dry-run` output was inspected for
  amd64, arm64 and both.

**Not verified — no image has been built for either architecture.** In
particular:

- whether the aarch64 `torch` wheel resolves the same working CUDA stack inside
  the container as the base image shipped. Modern torch pulls CUDA through
  separate `nvidia-*` dependency wheels, so wheel size and filename cannot
  answer this. Confirm with a real aarch64 build before relying on GPU serving
  there.
- the ESPnet dependency resolution, the DAC pre-warm step, and runtime
  behaviour on either architecture.

All runtime testing recorded elsewhere in this repository was done on H100
(sm90, amd64) **outside** Docker.

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
