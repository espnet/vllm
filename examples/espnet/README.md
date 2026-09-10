# ESPnet audio-LM examples

Tooling for the ESPnet audio language models supported by this fork:
**bagpiper** (Qwen3-8B backbone, text + audio generation with optional CFG),
**opuslm** (OLMo-2-7B backbone, TTS / ASR / text-LM), and
**opuslm_dialogue** (SmolLM2-1.7B backbone, spoken dialogue).

Layout:

- `convert/` — checkpoint converters (ESPnet / DeepSpeed → HF-style
  safetensors directories ready for `vllm serve`), plus
  `convert/bootstrap_assets.py`, which builds the `config.json` and tokenizer
  espnet does not publish from pinned public sources. The converters call it
  themselves, so converting an official checkpoint needs no other input. Run it
  directly to build only the assets, to list its pinned sources with URLs and
  hashes (`--print-sources`), or to diff what it generates against a directory
  you already trust (`--compare-with`).
- `serve_bagpiper.sh`, `serve_opuslm.sh`, `serve_opuslm_dialogue.sh` —
  server launch scripts (ports 9811 / 9812 / 9813 by default;
  `MODEL_PATH` is required, extra args pass through to `vllm serve`).
- `clients/` — reference clients for every supported task, plus a
  stress-test client for bagpiper. See `clients/README.md`.
- `docker/` — Dockerfile deriving from `vllm/vllm-openai:v0.28.0` with
  this fork's Python code and the ESPnet runtime dependencies.

See [Getting started (中文)](GETTING_STARTED.zh.md) for model conversion,
server configuration, client examples, and Docker deployment. Checkpoint
sources and generated assets are documented in [MODELS.md](MODELS.md).
