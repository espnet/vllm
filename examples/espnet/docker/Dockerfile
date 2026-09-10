# syntax=docker/dockerfile:1
# ESPnet serving image. build.sh pins the native vLLM v0.28 base by digest.
# Preserve its CUDA/PyTorch binaries; replace only the vLLM fork and add the
# explicitly versioned ESPnet serving package described in COMPATIBILITY.md.
ARG BASE_IMAGE=vllm/vllm-openai:v0.28.0
FROM ${BASE_IMAGE}
WORKDIR /workspace
COPY . /workspace/vllm-fork

# Preserve the upstream GPU stack while resolving additional dependencies.
RUN python3 /workspace/vllm-fork/examples/espnet/docker/check_dependencies.py \
    && python3 /workspace/vllm-fork/examples/espnet/docker/runtime_constraints.py \
        > /tmp/espnet-runtime-constraints.txt

# Repackage the matching upstream wheel's native extensions into this fork.
RUN uv pip install --python "$(command -v python3)" --no-cache \
        "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
        "setuptools-rust>=1.9.0" "packaging>=24.2" wheel jinja2 \
    && python3 -m pip download vllm==0.28.0 --no-deps -d /tmp/vllm-wheel \
    && python3 -m pip uninstall -y vllm \
    && VLLM_USE_PRECOMPILED=1 \
       VLLM_PRECOMPILED_WHEEL_LOCATION="$(ls /tmp/vllm-wheel/vllm-0.28.0*.whl)" \
       VLLM_DOCKER_BUILD_CONTEXT=1 \
       SETUPTOOLS_SCM_PRETEND_VERSION=0.28.0 \
       uv pip install --python "$(command -v python3)" --no-cache --no-build-isolation \
           -c /tmp/espnet-runtime-constraints.txt -e /workspace/vllm-fork \
    && rm -rf /tmp/vllm-wheel

# The source archive is unchanged upstream ESPnet 202609.post1. The preparation
# script changes only packaging constraints and gives the result a local
# version suffix; it does not rewrite an installed distribution's metadata.
ADD --checksum=sha256:a87f8aef6021236dcead56651c0b3c9f81d1feaec5fcbfe2bef0d22060ee298b \
    https://files.pythonhosted.org/packages/58/c9/7181d9cd1470bb09f497b5ee303655b184f5aa41d63b360f3d57b40e1314/espnet-202609.post1.tar.gz \
    /tmp/espnet.tar.gz
RUN mkdir /tmp/espnet-source \
    && tar -xzf /tmp/espnet.tar.gz -C /tmp/espnet-source --strip-components=1 \
    && python3 /workspace/vllm-fork/examples/espnet/docker/prepare_espnet.py /tmp/espnet-source \
    && uv pip install --python "$(command -v python3)" --no-cache -c /tmp/espnet-runtime-constraints.txt \
        '/tmp/espnet-source[asr,tts,kaldiio]' espnet_model_zoo==0.1.7 joblib scikit-learn \
    && python3 /workspace/vllm-fork/examples/espnet/docker/check_dependencies.py \
    && rm -rf /tmp/espnet-source /tmp/espnet.tar.gz

# Bake the DAC decoder used by both OpusLM variants. This is not a full LM
# checkpoint: users continue to mount their converted checkpoints at runtime.
RUN python3 -c "from espnet_model_zoo.downloader import ModelDownloader; \
    ModelDownloader().download_and_unpack('ftshijt/espnet_codec_dac_large_v1.4_360epoch')"

ENV VLLM_USE_V2_MODEL_RUNNER=0
LABEL org.opencontainers.image.source="https://github.com/espnet/vllm" \
      org.opencontainers.image.description="vLLM serving for Bagpiper, OpusLM and OpusLM-dialogue" \
      io.espnet.runtime.version="202609.post1+vllm.0.28.0"

# All published images must pass the CPU checks during the build itself.
RUN python3 /workspace/vllm-fork/examples/espnet/docker/smoke_test.py \
    && VLLM_TARGET_DEVICE=cpu vllm serve --help >/dev/null
# The upstream ENTRYPOINT remains ["vllm", "serve"].
