# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the installed container runtime without a GPU or checkpoint download."""

import importlib
import os
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    import torch

    import vllm

    expected_root = Path("/workspace/vllm-fork/vllm").resolve()
    actual_file = Path(vllm.__file__).resolve()
    if not actual_file.is_relative_to(expected_root):
        raise RuntimeError(f"vLLM is not imported from this fork: {actual_file}")
    for package, expected in (
        ("vllm", "0.28.0"),
        ("torch", "2.13.0"),
        ("torchvision", "0.28.0"),
        ("torchaudio", "2.11.0"),
    ):
        installed = version(package)
        if installed.split("+", 1)[0] != expected:
            raise RuntimeError(f"{package}: expected {expected}, got {installed}")
        print(f"{package}={installed}")
    if torch.version.cuda is None:
        raise RuntimeError("The image contains a CPU-only PyTorch build")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError("ESPnet audio generation requires the V1 model runner")

    for module in (
        "vllm._C",
        "torchaudio",
        "torchvision",
        "vllm.model_executor.models.bagpiper",
        "vllm.model_executor.models.opuslm",
        "vllm.model_executor.models.opuslm_dialogue",
        "vllm.tokenizers.opuslm",
        "vllm.tokenizers.opuslm_dialogue",
        "espnet2.bin.gan_codec_inference",
        "espnet2.tasks.ssl",
        "espnet_model_zoo.downloader",
        "joblib",
        "sklearn",
    ):
        importlib.import_module(module)
        print(f"import {module}: OK")

    from transformers import XcodecModel

    print(f"Xcodec decoder: {XcodecModel.__name__}")
    print(f"CUDA build: {torch.version.cuda}")
    print("CPU smoke checks passed; GPU inference still needs separate validation.")


if __name__ == "__main__":
    main()
