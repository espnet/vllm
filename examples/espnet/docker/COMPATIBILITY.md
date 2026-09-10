# ESPnet serving package compatibility

This image uses the official vLLM 0.28 CUDA runtime and an explicitly versioned
ESPnet serving package, `espnet==202609.post1+vllm.0.28.0`.

The PyPI ESPnet training distribution declares `torch>=2.9.1,<2.12` and
`setuptools<74`. vLLM 0.28 requires torch 2.13 and setuptools 77 or newer on
Python 3.12. Installing both unchanged cannot satisfy the dependency resolver.
The earlier Dockerfile forced torch back to 2.13 after installing ESPnet, which
left that declared conflict in the environment.

`prepare_espnet.py` starts from the checksum-pinned ESPnet 202609.post1 source
archive. It pins torch 2.13.0, torchaudio 2.11.0 and sentencepiece 0.2.2,
updates the setuptools constraint to vLLM's range, and adds the local version
suffix before building a new package. No ESPnet model implementation is changed. This is a serving
integration maintained in this repository, not an upstream ESPnet release or a
claim that the full ESPnet training suite supports this combination.

`runtime_constraints.py` preserves all installed base runtime versions, including
torch, CUDA libraries, Triton, NumPy, transformers, tokenizers and telemetry.
Build tools are resolved separately. Installation uses one
`uv` resolver for the ESPnet dependencies, including the official base's
`/etc/uv-overrides.txt`: upstream intentionally uses NCCL 2.30.7 for DeepEPv2,
although torch's wheel metadata pins NCCL 2.29.7 (see `docker/Dockerfile`).
`check_dependencies.py` runs `pip check` and permits exactly that one known
message with those exact installed versions. On ARM, NVIDIA’s cuSPARSELt
0.8.1 aarch64 wheel has the internal tag `manylinux2014_sbsa`, which pip does
not recognize. The check permits that one platform message only after verifying
the package version, internal tag, and actual shared library’s 64-bit little-endian
AArch64 ELF header. It does not rewrite wheel metadata. Any other error fails
before and after installation. There is no forced torch replacement. The source
and these scripts remain in `/workspace/vllm-fork/examples/espnet/docker` inside the image.

Publication requires the built image to pass dependency, import and CLI
checks. Actual inference for all three supported models has also been checked
on H100 and GB200; see [VALIDATION.md](VALIDATION.md). Repeat GPU validation
when changing this package variant or the CUDA runtime. CPU checks alone do
not establish CUDA ABI compatibility or audio generation quality.
