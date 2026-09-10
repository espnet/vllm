# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the base image's GPU stack and model frontend during installation."""

from importlib.metadata import distributions

PINNED_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "vllm",
    "numpy",
    "transformers",
    "tokenizers",
}

for distribution in sorted(distributions(), key=lambda d: d.metadata["Name"]):
    name = distribution.metadata["Name"].lower().replace("_", "-")
    if name in PINNED_PACKAGES or name.startswith("nvidia-"):
        print(f"{name}=={distribution.version}")
