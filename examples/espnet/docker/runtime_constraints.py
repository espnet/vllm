# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the complete base runtime while adding ESPnet dependencies."""

from importlib.metadata import distributions

BUILD_PACKAGES = {
    "pip",
    "setuptools",
    "setuptools-scm",
    "setuptools-rust",
    "packaging",
    "wheel",
}

versions = {}
for distribution in distributions():
    name = distribution.metadata["Name"].lower().replace("_", "-")
    if name not in BUILD_PACKAGES:
        # Debian's system packages may also exist later on sys.path. Preserve
        # the first (active) distribution, rather than pinning both versions.
        versions.setdefault(name, distribution.version)

for name, installed in sorted(versions.items()):
    print(f"{name}=={installed}")
