# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run pip check with narrowly verified exceptions in the pinned base image.

vLLM v0.28.0 requires newer NCCL for DeepEPv2 and installs it through
/etc/uv-overrides.txt (see docker/Dockerfile). Do not downgrade that binary
to satisfy torch's older metadata pin. All other dependency errors fail.
The ARM cuSPARSELt wheel also uses the nonstandard SBSA platform tag; verify
its actual ELF architecture before accepting that packaging discrepancy.
"""

import platform
import subprocess
import sys
from importlib.metadata import distribution, version


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
    allowed = {known}
    sbsa = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
    if platform.machine() == "aarch64" and sbsa in result.stdout.splitlines():
        package = distribution("nvidia-cusparselt-cu13")
        wheel = package.read_text("WHEEL") or ""
        binary = package.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0")
        with binary.open("rb") as stream:
            header = stream.read(20)
        if (
            package.version == "0.8.1"
            and "Tag: py3-none-manylinux2014_sbsa" in wheel.splitlines()
            and header[:6] == b"\x7fELF\x02\x01"
            and int.from_bytes(header[18:20], "little") == 183  # EM_AARCH64
        ):
            allowed.add(sbsa)
    errors = set(result.stdout.strip().splitlines())
    if (
        result.returncode != 1
        or not errors.issubset(allowed)
        or version("torch") != "2.13.0+cu130"
        or version("nvidia-nccl-cu13") != "2.30.7"
    ):
        raise SystemExit("Unexpected dependency errors; image validation failed")
    print("Only the pinned base image's verified packaging exceptions are present.")


if __name__ == "__main__":
    main()
