# ESPnet audio-LM docker image

Derives from `vllm/vllm-openai:v0.28.0`, swaps in this fork's Python code
(native extensions are reused from the official v0.28.0 wheel — the fork
is Python-only), and adds the ESPnet runtime dependencies for OpusLM
audio decode/encode.

Supports **x86_64/amd64** and **ARM64/aarch64** from a single Dockerfile.

For GitHub Actions builds and Docker Hub publication, see
[PUBLISHING.md](PUBLISHING.md). Both architectures build on native GitHub-hosted
runners. Adding the two Docker Hub secrets enables publication after successful
validation; forward [OWNER_SETUP.md](OWNER_SETUP.md) for the owner’s setup.

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

Both paths copy the current working tree into the image. Clone and check out a
specific commit before building when the exact source revision matters.

## Runtime and validation

The image preserves the pinned official CUDA/PyTorch binaries and installs
`espnet==202609.post1+vllm.0.28.0`, a serving package prepared from a
checksum-pinned source archive. Its model implementation is unchanged; its
packaging constraints accommodate the vLLM runtime. See
[COMPATIBILITY.md](COMPATIBILITY.md) for the precise changes and the upstream
NCCL override handled by the dependency check.

Every image must pass dependency validation, imports for all three models and
their codec/SSL dependencies, and CLI startup. GitHub repeats those checks
offline before publication. CPU checks establish neither GPU kernel execution
nor audio quality; see [VALIDATION.md](VALIDATION.md) for actual H100 and GB200
inference results.

Docker Hub publication is not established until a credentialed run pushes and
verifies the public tags. The workflow's successful build-only runs can be
inspected before the owner supplies those credentials.

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
