# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run pip check, allowing only the official base image's NCCL override.

vLLM v0.28.0 requires newer NCCL for DeepEPv2 and installs it through
/etc/uv-overrides.txt (see docker/Dockerfile). Do not downgrade that binary
to satisfy torch's older metadata pin. All other dependency errors fail.
"""

import subprocess
import sys
from importlib.metadata import version


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode == 0:
        return
    known = (
        "torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; "
        'platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.'
    )
    errors = result.stdout.strip().splitlines()
    if (
        result.returncode != 1
        or errors != [known]
        or version("torch") != "2.13.0+cu130"
        or version("nvidia-nccl-cu13") != "2.30.7"
    ):
        raise SystemExit("Unexpected dependency errors; image validation failed")
    print("Only the official vLLM base's documented NCCL override is present.")


if __name__ == "__main__":
    main()
