# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared preflight check for the ESPnet audio-LM serve scripts. Sourced, not
# executed.

# Fail if the `vllm` on PATH is not the build from this repository.
#
# A machine can easily carry a second vllm install (a system one under /opt, a
# stale wheel in the active environment). `vllm serve` then runs that build,
# which has no bagpiper/opuslm/opuslm_dialogue registration, and the server dies
# with a generic "architectures are not supported" line that points nowhere near
# the real cause. Checking here turns a confusing 60-second failure into a
# one-line one.
#
# find_spec does not execute vllm/__init__.py, so this costs an interpreter
# start and nothing else.
espnet_require_fork() {
  local repo_root="$1"
  local vllm_bin interp pkg_dir cand

  vllm_bin="$(command -v vllm || true)"
  if [[ -z "${vllm_bin}" ]]; then
    echo "ERROR: no 'vllm' on PATH." >&2
    echo "       Activate the environment this fork is installed into, e.g." >&2
    echo "       PATH=/path/to/venv/bin:\$PATH" >&2
    return 1
  fi

  # The console script's shebang names the interpreter that will import vllm.
  # It may be an absolute path or `/usr/bin/env python3`, so fall back through
  # the plain interpreter names when the first candidate answers nothing.
  interp="$(sed -n '1s|^#!\([^ ]*\).*|\1|p' -- "${vllm_bin}" 2>/dev/null || true)"
  pkg_dir=""
  for cand in "${interp}" python3 python; do
    [[ -n "${cand}" ]] || continue
    command -v "${cand}" >/dev/null 2>&1 || continue
    pkg_dir="$("${cand}" -c 'import importlib.util as u, os, sys
spec = u.find_spec("vllm")
sys.stdout.write(os.path.dirname(spec.origin) if spec and spec.origin else "")' 2>/dev/null || true)"
    [[ -n "${pkg_dir}" ]] && break
  done

  # All three models live in this fork, so any one of their files proves it.
  if [[ ! -f "${pkg_dir}/model_executor/models/opuslm.py" ]]; then
    echo "ERROR: the 'vllm' on PATH is not this fork." >&2
    echo "         entrypoint: ${vllm_bin}" >&2
    echo "       vllm package: ${pkg_dir:-<could not resolve>}" >&2
    echo "       Expected a build carrying model_executor/models/opuslm.py," >&2
    echo "       i.e. the one in ${repo_root}." >&2
    echo "       Install it (pip install -e .) and put that environment's bin" >&2
    echo "       directory first on PATH." >&2
    return 1
  fi
}
