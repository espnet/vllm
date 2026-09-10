# ESPnet audio-LM serving image.
#
# Derives from the official vllm/vllm-openai:v0.28.0 image (same binaries,
# same CUDA stack) and replaces the vLLM *Python* code with this fork,
# reusing the upstream precompiled v0.28.0 wheel for all native extensions
# (the fork's changes are Python-only). Then installs the ESPnet runtime
# dependencies needed for OpusLM audio decode (DAC) and encode (XEUS +
# kmeans).
#
# BUILD (from the repo root; the build context must be the repo root so
# the fork source can be COPY'd):
#   docker build -f examples/espnet/docker/Dockerfile -t espnet-vllm:v0.28.0 .
#
# Source options:
#   1. COPY of the local checkout (implemented below) — simple and
#      reproducible: the image contains exactly your working tree
#      (Dockerfile.dockerignore next to this file filters build artifacts
#      and .git; BuildKit prefers it over the repo-root .dockerignore).
#   2. git clone (alternative): replace the COPY line with
#        ARG VLLM_FORK_URL=https://example.com/your/vllm-fork.git
#        ARG VLLM_FORK_REF=espnet-audio-v0.28.0
#        RUN git clone --branch ${VLLM_FORK_REF} --depth 1 \
#            ${VLLM_FORK_URL} /workspace/vllm-fork
#      and build with --build-arg VLLM_FORK_URL=... VLLM_FORK_REF=...
#
# See README.md next to this file for run commands.
#
# ── ARCHITECTURES ────────────────────────────────────────────────────────
# One Dockerfile covers both x86_64/amd64 and ARM64/aarch64, because the
# base image is genuinely multi-arch. That was verified against the Docker
# Hub registry API on 2026-09-03, not assumed:
#
#   vllm/vllm-openai:v0.28.0
#     mediaType : application/vnd.docker.distribution.manifest.list.v2+json
#     index     : sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14
#     linux/amd64 sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635
#     linux/arm64 sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce
#
#   Each per-arch image config blob was fetched and reports architecture
#   "amd64"/"arm64" with os "linux", so the index is not merely claiming it.
#
# BASE_IMAGE lets you pin one architecture by digest for a reproducible
# single-arch build; build.sh next to this file does that for you. Leaving
# the default tag lets the daemon resolve the arch it is running on.
#
# Duplicating this file per architecture was rejected deliberately: the two
# copies would differ only in the FROM line and would drift. build.sh keeps
# the per-arch inputs explicit instead.
#
# See README.md for recorded build evidence and PUBLISHING.md for the
# dependency conflict that must be resolved before automatic publication.

ARG BASE_IMAGE=vllm/vllm-openai:v0.28.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

# ── Fork source ──────────────────────────────────────────────────────────
COPY . /workspace/vllm-fork

# ── Replace vLLM python with the fork ────────────────────────────────────
# VLLM_USE_PRECOMPILED=1 makes setup.py repackage native extensions (and
# the rust frontend) from an existing wheel instead of compiling.
# VLLM_PRECOMPILED_WHEEL_LOCATION pins that wheel to the official PyPI
# v0.28.0 release — exactly the binaries already in this base image —
# skipping the nightly-repo commit lookup (which would target upstream
# main, not the v0.28.0 base of this fork).
# VLLM_DOCKER_BUILD_CONTEXT=1 is what the upstream Dockerfile sets so
# setup.py does not assume a mutable .git.
# SETUPTOOLS_SCM_PRETEND_VERSION keeps the version deterministic and
# independent of git state in the build context.
# --no-build-isolation: the base image env already has the pinned torch;
# only the light build tools below are needed (no cmake/ninja under the
# precompiled path).
RUN python3 -m pip install --no-cache-dir \
        "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
        "setuptools-rust>=1.9.0" "packaging>=24.2" wheel jinja2 \
    && python3 -m pip download vllm==0.28.0 --no-deps -d /tmp/vllm-wheel \
    && python3 -m pip uninstall -y vllm \
    && VLLM_USE_PRECOMPILED=1 \
       VLLM_PRECOMPILED_WHEEL_LOCATION="$(ls /tmp/vllm-wheel/vllm-0.28.0*.whl)" \
       VLLM_DOCKER_BUILD_CONTEXT=1 \
       SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0 \
       python3 -m pip install --no-cache-dir --no-build-isolation \
           -e /workspace/vllm-fork \
    && rm -rf /tmp/vllm-wheel

# ── ESPnet runtime deps (OpusLM audio decode/encode) ─────────────────────
# espnet and espnet_model_zoo both drag in an older torch; install them
# first and force-reinstall the exact pins from vLLM v0.28.0's
# requirements/cuda.txt afterwards (verified necessary: a dry-run of
# either package reports "Would install torch-2.12.1" even with 2.13.0
# already satisfied).
# espnet_model_zoo is a hard requirement, not an optional extra:
# AudioCoding.from_pretrained() imports espnet_model_zoo.downloader to
# resolve an HF model tag, so without it every OpusLM /
# OpusLM-dialogue audio decode raises ModuleNotFoundError.
# joblib + scikit-learn: needed to load the XEUS kmeans model.
#
# ARCHITECTURE NOTE. This layer is the one place where the two architectures
# could diverge, because it force-reinstalls torch over the base image's copy.
# Checked on PyPI 2026-09-03: torch 2.13.0, torchvision 0.28.0 and torchaudio
# 2.11.0 all publish manylinux_2_28_aarch64 wheels as well as x86_64, and
# espnet/espnet_model_zoo ship pure-Python py3-none-any wheels, so nothing here
# is x86-only by availability.
#
# What that check does NOT establish: whether the aarch64 torch wheel resolves
# the same working CUDA stack inside the container as the base image shipped.
# Modern torch pulls CUDA through separate nvidia-* dependency wheels, so wheel
# size and filename cannot answer it. Confirm with a real aarch64 build before
# relying on GPU serving there.
RUN python3 -m pip install --no-cache-dir espnet espnet_model_zoo \
    && python3 -m pip install --no-cache-dir --force-reinstall \
        torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    && python3 -m pip install --no-cache-dir joblib scikit-learn

# ── Pre-warm the DAC codec into the image ────────────────────────────────
# OpusLM / OpusLM-dialogue fetch the DAC decoder through espnet_model_zoo,
# whose cache directory is NOT the HF hub cache: ModelDownloader() defaults
# to its own package directory when that is writable, which it is for root
# (verified: cachedir resolves to site-packages/espnet_model_zoo, 308 MB).
# So the `-v ~/.cache/huggingface` mount documented in README.md does not
# cover this one, and a container without egress would fail on the first
# audio decode. Bake it in instead.
#
# This pre-warms the tag that the shipped configs default to
# (dac_hf_model_tag). A checkpoint that overrides it still downloads at
# runtime and needs network on first audio decode.
RUN python3 -c "from espnet_model_zoo.downloader import ModelDownloader; \
    ModelDownloader().download_and_unpack('ftshijt/espnet_codec_dac_large_v1.4_360epoch')"

# ── Pin the V1 model runner ──────────────────────────────────────────────
# The audio hooks (per-stream sampling, CFG shadow merge, prefill stream
# replay, WAV egress) live in vllm/v1/worker/gpu_model_runner.py. v0.28.0
# routes dense models to the newer runner in vllm/v1/worker/gpu/model_runner.py
# by default, and that one has none of them, so a server started there boots
# fine and emits text while silently returning no audio.
#
# examples/espnet/serve_*.sh set this too, but a plain `docker run` goes
# straight to the inherited ENTRYPOINT and never reads those scripts, so the
# image itself has to carry the setting. Override it only if you are serving
# an unrelated model with this image.
ENV VLLM_USE_V2_MODEL_RUNNER=0

# ENTRYPOINT ["vllm", "serve"] is inherited from the base image; keep it
# so run commands stay compatible with the upstream image.
