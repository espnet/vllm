#!/usr/bin/env bash
# Serve an OpusLM-dialogue model with the vLLM OpenAI-compatible API.
#
# Dialogue prompts are built from the structured message list passed in
# mm_processor_kwargs.messages (roles system/user/assistant; the LAST
# message must be an empty assistant message marking the generation
# target); see clients/client_opuslm_dialogue.py for reference payloads.
#
# Usage:
#   MODEL_PATH=/path/to/OpusLM_dialogue bash serve_opuslm_dialogue.sh
#   MODEL_PATH=/path/to/OpusLM_dialogue PORT=9813 bash serve_opuslm_dialogue.sh
#
# NOTE: audio output requires BOTH espnet and espnet_model_zoo on the
# server. AudioCoding.from_pretrained() imports espnet_model_zoo.downloader
# to resolve the DAC model tag, so espnet alone raises ModuleNotFoundError
# on the first audio decode. espnet_model_zoo caches into its own package
# directory, not the HF hub cache. Audio input additionally pulls the XEUS
# checkpoint and kmeans model from HF (those do use the HF hub cache), and
# needs joblib + scikit-learn to load the kmeans model.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Required: keep the V1 model runner. The audio hooks (per-stream sampling,
# prefill stream replay, WAV egress) live in
# vllm/v1/worker/gpu_model_runner.py. v0.28.0 routes dense models to the newer
# runner in vllm/v1/worker/gpu/model_runner.py by default, which has none of
# them: the server would boot and emit text while silently returning no audio,
# so fail loudly instead of serving a half-working model.
if [[ "${VLLM_USE_V2_MODEL_RUNNER:-0}" != "0" ]]; then
  echo "ERROR: VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER} is unsupported." >&2
  echo "       The audio hooks exist only in the V1 model runner." >&2
  exit 1
fi
export VLLM_USE_V2_MODEL_RUNNER=0

# Async scheduling defaults to ON in v0.28.0. ASYNC_SCHEDULING=0 (our default)
# passes --no-async-scheduling, the configuration the audio path is verified
# on; set ASYNC_SCHEDULING=1 to take the upstream default instead.
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9813}"
MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-opuslm_dialogue}"
LOGFILE="${LOGFILE:-${SCRIPT_DIR}/logs/vllm_${PORT}.log}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH is not set. Usage:" >&2
  echo "  MODEL_PATH=/path/to/OpusLM_dialogue bash serve_opuslm_dialogue.sh [extra vllm args]" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOGFILE")"

EXTRA_ARGS=("$@")

# If user passes --port in CLI args, prefer that value.
HAS_PORT_ARG=0
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  if [[ "${arg}" == "--port" ]] && (( i + 1 < ${#EXTRA_ARGS[@]} )); then
    PORT="${EXTRA_ARGS[$((i + 1))]}"
    HAS_PORT_ARG=1
  elif [[ "${arg}" == --port=* ]]; then
    PORT="${arg#--port=}"
    HAS_PORT_ARG=1
  fi
done

# Preflight: the `vllm` on PATH has to be this fork, or the model is unknown.
# shellcheck source=examples/espnet/preflight.sh
source "${SCRIPT_DIR}/preflight.sh"
espnet_require_fork "$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Preflight: fail early with clear diagnostics if port is occupied.
if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: port ${PORT} is already in use." >&2
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >&2 || true
    exit 1
  fi
elif ! python3 -c "import socket; s = socket.socket(); s.bind(('', ${PORT})); s.close()" 2>/dev/null; then
  echo "ERROR: port ${PORT} is already in use." >&2
  exit 1
fi

PORT_ARG=()
if [[ "${HAS_PORT_ARG}" -eq 0 ]]; then
  PORT_ARG=(--port "${PORT}")
fi

ASYNC_ARG=()
if [[ "${ASYNC_SCHEDULING}" == "0" ]]; then
  ASYNC_ARG=(--no-async-scheduling)
fi

echo "[serve_opuslm_dialogue] VLLM_USE_V2_MODEL_RUNNER=0 ASYNC_SCHEDULING=${ASYNC_SCHEDULING}" >&2

# max-model-len 8192 matches the checkpoint's max_position_embeddings.
# audio limit 8: a dialogue prompt can carry a speaker prompt plus
# several user audio turns.
# `vllm serve` is the canonical entrypoint on v0.28.0.
vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --max-model-len 8192 \
    --host "$HOST" \
    "${PORT_ARG[@]}" \
    "${ASYNC_ARG[@]}" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1024 \
    --tensor-parallel-size 1 \
    --limit-mm-per-prompt '{"audio": 8}' \
    --enable-prefix-caching \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOGFILE"
