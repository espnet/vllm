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

for distribution in sorted(distributions(), key=lambda d: d.metadata["Name"]):
    name = distribution.metadata["Name"].lower().replace("_", "-")
    if name not in BUILD_PACKAGES:
        print(f"{name}=={distribution.version}")
