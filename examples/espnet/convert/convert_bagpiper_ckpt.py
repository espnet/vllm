# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert an ESPnet Bagpiper (SpeechLM) DeepSpeed checkpoint to vLLM format.

Takes the mp_rank_00_model_states.pt from an ESPnet training run and produces
a complete HuggingFace-style checkpoint directory with sharded safetensors,
config.json, tokenizer files, etc. ready for vLLM inference.

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
        candidate = input_path / "mp_rank_00_model_states.pt"
        if candidate.exists():
            input_path = candidate
        else:
            print(f"ERROR: {candidate} not found. Please provide the .pt file directly.")
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

    if args.dry_run:
        print("\nDry run complete. No files saved.")
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

    print(f"\nDone! vLLM checkpoint saved to: {output_dir}")
    print(f"Files: {sorted(os.listdir(output_dir))}")


if __name__ == "__main__":
    main()
