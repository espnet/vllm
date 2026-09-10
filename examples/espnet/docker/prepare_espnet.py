# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare an explicitly versioned ESPnet source variant for this CUDA image.

The upstream training distribution caps torch and setuptools below vLLM's
requirements. This serving-only variant changes those packaging constraints;
no ESPnet model implementation is modified. Validate the resulting runtime,
not just the dependency resolver, before publishing the image.
"""

import argparse
from pathlib import Path

UPSTREAM_VERSION = "202609.post1"
RUNTIME_VERSION = UPSTREAM_VERSION + "+vllm.0.28.0"


def prepare(source: Path) -> None:
    version_file = source / "version.txt"
    if version_file.read_text().strip() != UPSTREAM_VERSION:
        raise ValueError("Unexpected ESPnet source version")
    project = source / "pyproject.toml"
    text = project.read_text()
    replacements = {
        '"setuptools>=38.5.1,<74.0.0"': ('"setuptools>=77.0.3,<81.0.0"', 2),
        '"torch>=2.9.1,<2.12"': ('"torch==2.13.0"', 1),
        '"torchaudio>=2.9.1,<2.12"': ('"torchaudio==2.11.0"', 1),
        '"sentencepiece==0.2.1"': ('"sentencepiece==0.2.2"', 1),
    }
    for old, (new, expected_count) in replacements.items():
        if text.count(old) != expected_count:
            raise ValueError(f"Unexpected ESPnet dependency declaration: {old}")
        text = text.replace(old, new)
    project.write_text(text)
    version_file.write_text(RUNTIME_VERSION + "\n")
    print(f"Prepared ESPnet serving variant {RUNTIME_VERSION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    prepare(parser.parse_args().source)
