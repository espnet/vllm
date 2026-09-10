# ESPnet serving package compatibility

This image uses the official vLLM 0.28 CUDA runtime and an explicitly versioned
ESPnet serving package, `espnet==202609.post1+vllm.0.28.0`.

The PyPI ESPnet training distribution declares `torch>=2.9.1,<2.12` and
`setuptools<74`. vLLM 0.28 requires torch 2.13 and setuptools 77 or newer on
Python 3.12. Installing both unchanged cannot satisfy the dependency resolver.
The earlier Dockerfile forced torch back to 2.13 after installing ESPnet, which
left that declared conflict in the environment.

`prepare_espnet.py` starts from the checksum-pinned ESPnet 202609.post1 source
archive. It pins torch 2.13.0 and torchaudio 2.11.0, updates the setuptools
constraint to vLLM's range, and adds the local version suffix before building
a new package. No ESPnet model implementation is changed. This is a serving
integration maintained in this repository, not an upstream ESPnet release or a
claim that the full ESPnet training suite supports this combination.

`runtime_constraints.py` preserves the base image's torch, CUDA libraries,
Triton, NumPy, transformers and tokenizer versions. Installation uses one
resolver for the ESPnet dependencies and runs `pip check`; there is no forced
torch replacement or ignored dependency error. The source and these scripts
remain in `/workspace/vllm-fork/examples/espnet/docker` inside the image.

Publication requires the built image to pass dependency, import and CLI
checks. Record actual GPU inference for all three supported models before
promoting this package variant for the first time. CPU checks alone do not
establish CUDA ABI compatibility or audio generation quality.
