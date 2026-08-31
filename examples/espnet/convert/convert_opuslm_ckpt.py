# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert an ESPnet OpusLM checkpoint to vLLM format.

Works for both ``opuslm`` and ``opuslm_dialogue`` checkpoints (they share the
same ESPnet module layout; no model-specific logic is needed).

Input is either an ESPnet ``model.pth`` (a flat ``name -> tensor`` state
dict) or a DeepSpeed checkpoint dict carrying the state dict under a
``"module"`` key. The checkpoint is loaded with ``weights_only=True`` and
``mmap=True`` so the full model is never materialized in RAM at once.

Key mapping (ESPnet -> vLLM/HF):
    corelm.emb.weight            -> model.embed_tokens.weight
    corelm.decoders.model.<rest> -> model.<rest>          (prefix strip)
    corelm.lm_head.weight        -> lm_head.weight
    corelm.head_emb.weight       -> head_emb.weight
    criterion.*                  -> dropped
Any other key is an error (listing the offending keys) so silent weight
loss is impossible.

Usage:
    python convert_opuslm_ckpt.py <model.pth> <output_dir> \
        --ref-dir /path/to/reference/opuslm-checkpoint

    # Inspect the key mapping without writing anything:
    python convert_opuslm_ckpt.py <model.pth> <output_dir> \
        --ref-dir /path/to/ref --dry-run
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

# Non-weight files to copy from the reference checkpoint (when present).
# OpusLM checkpoints have no generation_config.json / chat_template.jinja;
# the chat template is inline in tokenizer_config.json.
CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]

MAX_SHARD_BYTES = 5 * 1024**3  # 5 GB

# Exact-name mappings, checked before the prefix rules.
EXACT_KEY_MAP = {
    "corelm.emb.weight": "model.embed_tokens.weight",
    "corelm.lm_head.weight": "lm_head.weight",
    "corelm.head_emb.weight": "head_emb.weight",
}

DECODER_PREFIX = "corelm.decoders.model."
DROP_PREFIX = "criterion."


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
    """Load an ESPnet/DeepSpeed checkpoint and extract the state dict.

    weights_only=True + mmap=True: tensors stay backed by the file until
    they are serialized shard by shard, keeping peak RSS low.
    """
    print(f"Loading checkpoint from {input_path} ...")
    ckpt = torch.load(
        input_path, map_location="cpu", weights_only=True, mmap=True
    )

    if isinstance(ckpt, dict) and "module" in ckpt:
        state_dict = ckpt["module"]
        print("Extracted 'module' key from DeepSpeed checkpoint.")
    elif isinstance(ckpt, dict) and all(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        state_dict = ckpt
        print("Checkpoint is a plain state dict.")
    else:
        raise ValueError(
            f"Unexpected checkpoint format. Top-level keys: {list(ckpt.keys())[:20]}"
        )

    print(f"Loaded {len(state_dict)} tensors")
    return state_dict


def map_keys(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[tuple[str, str]], list[str]]:
    """Apply the ESPnet -> vLLM key mapping.

    Returns (mapped_state_dict, mapping_log, dropped_keys). Raises on any
    key that matches none of the rules, listing all offenders.
    """
    mapped: dict[str, torch.Tensor] = {}
    mapping_log: list[tuple[str, str]] = []
    dropped: list[str] = []
    unexpected: list[str] = []

    for name, tensor in state_dict.items():
        if name in EXACT_KEY_MAP:
            new_name = EXACT_KEY_MAP[name]
        elif name.startswith(DECODER_PREFIX):
            new_name = "model." + name[len(DECODER_PREFIX):]
        elif name.startswith(DROP_PREFIX):
            dropped.append(name)
            continue
        else:
            unexpected.append(name)
            continue

        if new_name in mapped:
            raise ValueError(
                f"Key collision: both map to {new_name!r} "
                f"(second source: {name!r})"
            )
        mapped[new_name] = tensor
        mapping_log.append((name, new_name))

    if unexpected:
        raise ValueError(
            "Unexpected checkpoint keys (no mapping rule matches). "
            "Refusing to convert to avoid silent weight loss:\n  "
            + "\n  ".join(unexpected)
        )

    return mapped, mapping_log, dropped


def save_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    output_dir: str,
    max_shard_bytes: int = MAX_SHARD_BYTES,
) -> None:
    """Save state_dict as sharded safetensors files with an index file.

    Same sharding scheme as convert_bagpiper_ckpt.py: greedy accumulation
    in dict order; a single shard is named model.safetensors with no index.
    """
    tensors_with_size = []
    for name, tensor in state_dict.items():
        size_bytes = tensor.numel() * tensor.element_size()
        tensors_with_size.append((name, tensor, size_bytes))

    total_size = sum(s for _, _, s in tensors_with_size)
    print(f"Total model size: {total_size / 1024**3:.2f} GB")

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
        # .contiguous() is a no-op for already-contiguous tensors; it only
        # guards against odd strides from mmap-backed storages.
        save_file({k: v.contiguous() for k, v in shard.items()}, filepath)
        shard_size = sum(t.numel() * t.element_size() for t in shard.values())
        print(f"  {filename} ({shard_size / 1024**3:.2f} GB, {len(shard)} tensors)")

        for name in shard:
            weight_map[name] = filename

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert an ESPnet OpusLM / OpusLM-dialogue checkpoint "
            "(model.pth) to vLLM format."
        )
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to model.pth (or a directory containing it)",
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
        help="Only print the key mapping, don't save anything",
    )
    args = parser.parse_args()

    # Resolve input path: accept either the .pth file or its parent directory
    input_path = Path(args.input)
    if input_path.is_dir():
        candidate = input_path / "model.pth"
        if candidate.exists():
            input_path = candidate
        else:
            print(f"ERROR: {candidate} not found. Please provide the .pth file directly.")
            sys.exit(1)

    if not input_path.exists():
        print(f"ERROR: {input_path} does not exist.")
        sys.exit(1)

    ref_dir = Path(args.ref_dir)
    if not ref_dir.exists():
        print(f"ERROR: Reference checkpoint dir {ref_dir} does not exist.")
        sys.exit(1)

    state_dict = load_checkpoint(str(input_path))
    mapped, mapping_log, dropped = map_keys(state_dict)

    print(f"\nKey mapping ({len(mapping_log)} kept, {len(dropped)} dropped):")
    for old, new in mapping_log:
        print(f"  {old}  ->  {new}")
    for name in dropped:
        print(f"  {name}  ->  (dropped)")

    if args.dry_run:
        print("\nDry run complete. No files saved.")
        return

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print()
    max_shard_bytes = parse_size(args.max_shard_size)
    save_sharded_safetensors(mapped, output_dir, max_shard_bytes)

    copy_config_files(output_dir, ref_dir)

    print(f"\nDone! vLLM checkpoint saved to: {output_dir}")
    print(f"Files: {sorted(os.listdir(output_dir))}")


if __name__ == "__main__":
    main()
