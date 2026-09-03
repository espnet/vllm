# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert a released ESPnet Bagpiper checkpoint to a vLLM-loadable directory.

WHY THIS IS NEEDED
------------------
The published Bagpiper repositories are NOT vLLM-loadable as they stand. As of
2026-09-03, espnet/bagpiper-sft contains:

    model.pt                        native ESPnet weights, {"module": state_dict}
    train_stage3_qwen3_base.yaml    model configuration
    inference_{audio,text}.yaml     ESPnet decoding configs
    MANIFEST.json, SHA256SUMS, LICENSE, THIRD_PARTY_NOTICES.md, README.md

There is no config.json, no tokenizer, and no safetensors -- its own model card
says "It is not a Transformers from_pretrained directory and no vLLM
compatibility is claimed." espnet/bagpiper (the pre-trained base) is the same
shape with base.pt. So a real conversion step is required; this script is it.

INPUT
    Either the released model.pt, or the raw DeepSpeed
    mp_rank_00_model_states.pt it was built from (espnet/bagpiper-sft's
    MANIFEST.json records that provenance), or a directory containing either.
    Both carry the state dict under a "module" key.

WEIGHT HANDLING
    Weight names are written to safetensors verbatim; all renaming to vLLM
    module names happens at load time in
    ``BagpiperForConditionalGeneration.hf_to_vllm_mapper``. Because of that,
    a key the mapper does not recognise would be dropped silently at load
    time, so every tensor is checked against KNOWN_PREFIXES first and the
    script exits non-zero listing anything unexpected. ``vocab_weight`` is
    dropped explicitly and loudly (see DROP_KEYS). dtypes are reported before
    and after so a silent cast cannot hide.

CONFIG AND TOKENIZER
    These are not published by espnet, so they are copied from --ref-dir: any
    directory that already holds a Bagpiper config.json plus the tokenizer
    files. config.json is then rewritten to the canonical naming
    (``model_type: "bagpiper"``, ``architectures:
    ["BagpiperForConditionalGeneration"]``) even when the reference still uses
    the legacy ``speechlm`` names.

    For reference, the vocabulary layout is fully determined by the released
    YAML: tokenizer Qwen/Qwen3-8B-Base (151,936 tokens) sits at
    [text_token_offset=256, text_token_end=152192), the 8 Xcodec streams
    occupy [152192, 152192 + 8*1025) and vocab_size is 160,392.

Weight keys are written to safetensors verbatim (no renames); all mapping to
vLLM module names happens at load time in the model
(``BagpiperForConditionalGeneration.hf_to_vllm_mapper``).

After copying the config/tokenizer files from ``--ref-dir``, config.json in
the output directory is rewritten to the new naming
(``model_type: "bagpiper"``, ``architectures:
["BagpiperForConditionalGeneration"]``) so that new conversions always emit
the canonical names even when the reference checkpoint still uses the legacy
``speechlm`` naming.

Usage:
    python convert_bagpiper_ckpt.py <input_path> <output_dir> \
        --ref-dir /path/to/reference/bagpiper-checkpoint

Example:
    python convert_bagpiper_ckpt.py \
        /path/to/exp/.../mp_rank_00_model_states.pt \
        /path/to/output/bagpiper-step272500 \
        --ref-dir /path/to/hf/vLLM_alm/bagpiper
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# Non-weight files to copy from the reference checkpoint
CONFIG_FILES = [
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
]

MAX_SHARD_BYTES = 5 * 1024**3  # 5 GB

# Weight-name prefixes the model knows how to load. These are the source side
# of BagpiperForConditionalGeneration.hf_to_vllm_mapper; the converter writes
# keys verbatim and the model renames them at load time, so anything outside
# this set would be silently dropped by the loader instead of failing.
#
# Counts observed in espnet/bagpiper-sft (model.pt, 1382 tensors):
#   model.layers.                                            396
#   multimodal_io_dict.continuous_audio.model.audio_tower.   525
#   multimodal_io_dict.discrete_audio.codec_model.           454
#   adaptor.continuous_audio.                                  2
#   model.embed_tokens. / model.norm. / lm_head. / stream_emb. 1 each
KNOWN_PREFIXES = (
    "model.layers.",
    "model.norm.",
    "model.embed_tokens.",
    "lm_head.",
    "multimodal_io_dict.continuous_audio.model.audio_tower.",
    "adaptor.continuous_audio.",
    "stream_emb.",
    "multimodal_io_dict.discrete_audio.codec_model.",
)

# Tensors that are deliberately not carried into the vLLM checkpoint, with the
# reason. Dropping is explicit and logged -- never silent.
DROP_KEYS = {
    "vocab_weight": (
        "per-token loss weighting vector, fp32 shape (vocab_size,). Present in "
        "espnet/bagpiper-sft because the strict ESPnet loader requires it; it "
        "is not a model parameter and vLLM never reads it."
    ),
}

# Filenames that may hold the state dict, in the order we look for them inside
# a directory. model.pt is what espnet/bagpiper-sft publishes;
# mp_rank_00_model_states.pt is the raw DeepSpeed name it was built from.
CKPT_FILENAMES = ("model.pt", "mp_rank_00_model_states.pt")


def parse_size(size_str: str) -> int:
    """Parse a human-readable size string like '5GB' into bytes."""
    size_str = size_str.strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(GB|MB|KB|B)?$", size_str)
    if not match:
        raise ValueError(f"Invalid size string: {size_str}")
    value = float(match.group(1))
    unit = match.group(2) or "B"
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return int(value * multipliers[unit])


def load_checkpoint(input_path: str) -> dict[str, torch.Tensor]:
    """Load a DeepSpeed checkpoint and extract the model state dict."""
    print(f"Loading checkpoint from {input_path} ...")
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if "module" in ckpt:
        state_dict = ckpt["module"]
        print("Extracted 'module' key from DeepSpeed checkpoint.")
    elif isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        state_dict = ckpt
        print("Checkpoint is already a plain state dict.")
    else:
        raise ValueError(
            f"Unexpected checkpoint format. Top-level keys: {list(ckpt.keys())[:20]}"
        )

    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {len(state_dict)} tensors, {total_params:,} params")
    return state_dict


def validate_and_filter(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int], list[str]]:
    """Drop the known non-parameters, then insist every survivor is loadable.

    Returns (kept, per-prefix counts, dropped key names). Raises SystemExit
    listing the offending keys if anything falls outside KNOWN_PREFIXES, so a
    checkpoint whose layout has moved fails here instead of producing a
    directory that loads with quietly missing weights.
    """
    kept: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {p: 0 for p in KNOWN_PREFIXES}
    dropped: list[str] = []
    unknown: list[str] = []

    for name, tensor in state_dict.items():
        if name in DROP_KEYS:
            dropped.append(name)
            continue
        for prefix in KNOWN_PREFIXES:
            if name.startswith(prefix):
                counts[prefix] += 1
                kept[name] = tensor
                break
        else:
            unknown.append(name)

    if unknown:
        print(
            f"\nERROR: {len(unknown)} tensor(s) match no prefix the model can "
            f"load, so they would be silently ignored at load time:",
            file=sys.stderr,
        )
        for name in unknown[:20]:
            print(f"  {name}", file=sys.stderr)
        if len(unknown) > 20:
            print(f"  ... and {len(unknown) - 20} more", file=sys.stderr)
        print(
            "\nEither the checkpoint layout changed or this is not a Bagpiper "
            "checkpoint. Add the prefix to KNOWN_PREFIXES (and to the model's "
            "hf_to_vllm_mapper) if it is genuinely new; do not ignore this.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    empty = [p for p, c in counts.items() if c == 0]
    if empty:
        print(
            f"\nERROR: no tensors found for {len(empty)} expected weight "
            f"group(s): {empty}",
            file=sys.stderr,
        )
        print(
            "A Bagpiper checkpoint carries all of them (LLM body, audio tower, "
            "adaptor, stream embedding, Xcodec decoder). Converting anyway "
            "would produce a model that loads but cannot run.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    for name in dropped:
        print(f"  dropping {name}: {DROP_KEYS[name]}")
    return kept, counts, dropped


def report_dtypes(state_dict: dict[str, torch.Tensor], label: str) -> dict[str, int]:
    """Print the dtype histogram. Called before and after so any cast shows."""
    hist: dict[str, int] = {}
    for tensor in state_dict.values():
        key = str(tensor.dtype)
        hist[key] = hist.get(key, 0) + 1
    pretty = ", ".join(f"{k}: {v}" for k, v in sorted(hist.items()))
    print(f"  dtypes {label}: {pretty}")
    return hist


def save_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
    max_shard_bytes: int = MAX_SHARD_BYTES,
) -> None:
    """Save state_dict as sharded safetensors files with an index file."""
    tensors_with_size = []
    for name, tensor in state_dict.items():
        size_bytes = tensor.numel() * tensor.element_size()
        tensors_with_size.append((name, tensor, size_bytes))

    total_size = sum(s for _, _, s in tensors_with_size)
    print(f"Total model size: {total_size / 1024**3:.2f} GB")

    # Build shards
    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0

    for name, tensor, size_bytes in tensors_with_size:
        if current_size + size_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += size_bytes

    if current_shard:
        shards.append(current_shard)

    num_shards = len(shards)
    print(f"Saving {num_shards} shard(s) ...")

    weight_map = {}
    metadata = {"total_size": total_size}

    for shard_idx, shard in enumerate(shards):
        shard_num = shard_idx + 1
        if num_shards == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{shard_num:05d}-of-{num_shards:05d}.safetensors"

        filepath = os.path.join(output_dir, filename)
        save_file(shard, filepath)
        shard_size = sum(t.numel() * t.element_size() for t in shard.values())
        print(f"  {filename} ({shard_size / 1024**3:.2f} GB, {len(shard)} tensors)")

        for name in shard:
            weight_map[name] = filename

    # Write index file
    if num_shards > 1:
        index = {"metadata": metadata, "weight_map": weight_map}
        index_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print("  Saved index: model.safetensors.index.json")


def copy_config_files(output_dir: str, ref_dir: Path) -> None:
    """Copy config and tokenizer files from the reference checkpoint."""
    print(f"\nCopying config/tokenizer files from {ref_dir} ...")
    for fname in CONFIG_FILES:
        src = ref_dir / fname
        dst = os.path.join(output_dir, fname)
        if src.exists():
            shutil.copy2(str(src), dst)
            print(f"  Copied {fname}")
        else:
            print(f"  WARNING: {fname} not found in reference dir, skipping")


def rewrite_model_naming(output_dir: str) -> None:
    """Rewrite config.json (and tokenizer_config.json) to the new naming.

    Sets ``architectures=["BagpiperForConditionalGeneration"]`` and
    ``model_type="bagpiper"`` in config.json, leaving every other key
    untouched. If tokenizer_config.json carries a ``model_type`` key
    (the shipped checkpoint does), patch it too.
    """
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["architectures"] = ["BagpiperForConditionalGeneration"]
        config["model_type"] = "bagpiper"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        print(
            "  Rewrote config.json: model_type=bagpiper, "
            "architectures=[BagpiperForConditionalGeneration]"
        )

    tok_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tok_config_path):
        with open(tok_config_path) as f:
            tok_config = json.load(f)
        if "model_type" in tok_config:
            tok_config["model_type"] = "bagpiper"
            with open(tok_config_path, "w") as f:
                json.dump(tok_config, f, indent=2)
                f.write("\n")
            print("  Rewrote tokenizer_config.json: model_type=bagpiper")


def print_weight_summary(state_dict: dict[str, torch.Tensor]) -> None:
    """Print a summary of weight groups."""
    groups: dict[str, int] = {}
    for name in sorted(state_dict.keys()):
        parts = name.split(".")
        prefix = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups[prefix] = groups.get(prefix, 0) + 1

    print("\nWeight groups:")
    for prefix, count in sorted(groups.items()):
        print(f"  {prefix}: {count} tensors")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ESPnet Bagpiper (SpeechLM) checkpoint to vLLM format."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to mp_rank_00_model_states.pt (or directory containing it)",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for the vLLM checkpoint",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help="Maximum shard size (default: 5GB)",
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        required=True,
        help="Reference checkpoint dir for config/tokenizer files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print weight summary, don't save",
    )
    args = parser.parse_args()

    # Resolve input path: accept either the .pt file or its parent directory
    input_path = Path(args.input)
    if input_path.is_dir():
        for candidate_name in CKPT_FILENAMES:
            candidate = input_path / candidate_name
            if candidate.exists():
                input_path = candidate
                break
        else:
            print(
                f"ERROR: none of {list(CKPT_FILENAMES)} found in {input_path}. "
                f"Pass the checkpoint file directly."
            )
            sys.exit(1)

    if not input_path.exists():
        print(f"ERROR: {input_path} does not exist.")
        sys.exit(1)

    ref_dir = Path(args.ref_dir)
    if not ref_dir.exists():
        print(f"ERROR: Reference checkpoint dir {ref_dir} does not exist.")
        sys.exit(1)

    # Load checkpoint
    state_dict = load_checkpoint(str(input_path))
    print_weight_summary(state_dict)

    print("\nValidating weight coverage ...")
    dtypes_in = report_dtypes(state_dict, "in checkpoint")
    state_dict, prefix_counts, dropped = validate_and_filter(state_dict)
    print(f"  kept {len(state_dict)} tensor(s), dropped {len(dropped)}")
    for prefix, count in prefix_counts.items():
        print(f"    {count:>5}  {prefix}*")

    if args.dry_run:
        print("\nDry run complete. Validation passed. No files saved.")
        return

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    print()
    max_shard_bytes = parse_size(args.max_shard_size)
    save_sharded_safetensors(state_dict, output_dir, max_shard_bytes)

    # Copy config and tokenizer files, then rewrite to the new naming
    copy_config_files(output_dir, ref_dir)
    rewrite_model_naming(output_dir)

    dtypes_out = report_dtypes(state_dict, "written")
    if dtypes_out != {k: v for k, v in dtypes_in.items() if v and k in dtypes_out} \
            and set(dtypes_out) - set(dtypes_in):
        print(
            "  WARNING: a dtype appears in the output that was not in the "
            "input; check for an unintended cast."
        )

    print(f"\nDone. vLLM checkpoint written to: {output_dir}")
    print(f"  tensors      : {len(state_dict)}")
    print(f"  dropped      : {len(dropped)} ({', '.join(dropped) if dropped else 'none'})")
    print(f"  files        : {sorted(os.listdir(output_dir))}")
    print("\nServe it with:")
    print(f"  MODEL_PATH={output_dir} bash examples/espnet/serve_bagpiper.sh")
    print("Then send a request (describe the scene, quote any spoken line):")
    print("  python examples/espnet/clients/client_bagpiper.py --task tts --out demo.wav")


if __name__ == "__main__":
    main()
