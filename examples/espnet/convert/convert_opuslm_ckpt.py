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

CONFIG AND TOKENIZER
    espnet publishes neither, so by default they are **built from public
    pinned sources** by bootstrap_assets.py. Both are fully determined by
    artifacts you can download:

    * the tokenizer is the backbone's, copied byte-for-byte from the repo the
      released ``config.yaml`` names in ``subword_model``
      (allenai/OLMo-2-1124-7B for opuslm, HuggingFaceTB/SmolLM-1.7B for
      opuslm_dialogue), with only the chat template replaced -- see
      fix_chat_template below for why;
    * every vocabulary boundary comes from that YAML's ``token_bias`` block and
      every special-token id is looked up **by name** in its ``token_list``,
      so a checkpoint whose special block moved fails loudly instead of
      silently mapping to the wrong token;
    * the transformer geometry comes from the backbone named in
      ``transformer_conf.hf_model_tag`` (which for opuslm_dialogue is a
      *different* repo than the tokenizer: SmolLM2-1.7B-Instruct).

    So no pre-existing converted directory is needed. ``--ref-dir`` is still
    accepted for reusing one you already have.

Usage:
    # from the official checkpoint, nothing else needed:
    python convert_opuslm_ckpt.py <model.pth> <output_dir> --model opuslm

    # Inspect the key mapping without writing anything:
    python convert_opuslm_ckpt.py <model.pth> <output_dir> --dry-run

Example:
    hf download espnet/OpusLM_7B_Anneal --local-dir ~/models/opuslm
    python convert_opuslm_ckpt.py ~/models/opuslm ~/models/opuslm-vllm \
        --model opuslm

    hf download espnet/multi_turn_SDS_RLAIF --local-dir ~/models/sds
    python convert_opuslm_ckpt.py ~/models/sds/2epoch.pth \
        ~/models/opuslm-dialogue-vllm --model opuslm_dialogue
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_assets  # noqa: E402  (same directory, not an installed module)

# Filenames that may hold the state dict, in the order we look for them inside
# a directory. espnet/OpusLM_7B_Anneal publishes model.pth;
# espnet/multi_turn_SDS_RLAIF publishes 2epoch.pth.
CKPT_FILENAMES = ("model.pth", "2epoch.pth")

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


def fix_chat_template(output_dir: str) -> None:
    """Replace the checkpoint's chat template with a content-only one.

    ESPnet checkpoints ship a template that wraps each message in `<|role|>`
    markers. Those markers are not entries of OpusLM's inner BPE vocabulary, so
    the tokenizer encodes them as ordinary text and the model receives several
    tokens of noise it was never trained on. All the structure OpusLM does need
    -- `<sos/eos>`, the task token, `<text_bpe_start/end>`, the ARDelay pads and
    `<codec_ssl_start/end>` -- is added afterwards by OpusLMTokenizer, in the
    reserved ID range below text_token_start where it belongs.

    Operators who want a different rendering can still pass `--chat-template`
    to `vllm serve`, which overrides what is written here.
    """
    path = os.path.join(output_dir, "tokenizer_config.json")
    if not os.path.exists(path):
        print("  WARNING: no tokenizer_config.json to fix the chat template in")
        return

    with open(path) as f:
        cfg = json.load(f)

    cfg["chat_template"] = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("  Rewrote chat_template in tokenizer_config.json (content only)")


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

    fix_chat_template(output_dir)


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
        "--model",
        choices=("opuslm", "opuslm_dialogue"),
        default=None,
        help=(
            "Which of the two models this checkpoint is. Required unless "
            "--ref-dir is given, because it selects the released config.yaml "
            "and backbone the config/tokenizer are built from."
        ),
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        default=None,
        help=(
            "Reuse the config/tokenizer from an existing converted directory "
            "instead of building them from the pinned public sources. Only "
            "needed if you already have one."
        ),
    )
    parser.add_argument(
        "--assets-from",
        type=Path,
        default=None,
        help=(
            "Build the config/tokenizer from pinned source files already on "
            "disk instead of downloading them (see "
            "'bootstrap_assets.py --model opuslm --print-sources')"
        ),
    )
    parser.add_argument(
        "--no-legacy-overrides",
        action="store_true",
        help=(
            "Emit purely backbone-derived config values instead of the ones "
            "the validated reference config uses. Changes model numerics; "
            "read what bootstrap_assets.py prints before using it."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the key mapping, don't save anything",
    )
    args = parser.parse_args()

    if args.model is None and args.ref_dir is None and not args.dry_run:
        parser.error(
            "pass --model opuslm or --model opuslm_dialogue (it selects the "
            "released config.yaml and backbone to build the config/tokenizer "
            "from), or --ref-dir to reuse an existing converted directory"
        )

    # Resolve input path: accept either the .pth file or its parent directory
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

    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    if ref_dir is not None and not ref_dir.exists():
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

    # Config and tokenizer first: they are small, and building them can fail on
    # a network or hash problem. Better to find that out before writing 14 GB.
    if ref_dir is not None:
        copy_config_files(output_dir, ref_dir)
    else:
        print("\nBuilding config/tokenizer from pinned public sources ...")
        # Reported so the generator can flag a mismatch with what the training
        # YAML asked for. It does not pick config.torch_dtype from this.
        dtypes = {str(t.dtype).removeprefix("torch.") for t in mapped.values()}
        checkpoint_dtype = next(iter(dtypes)) if len(dtypes) == 1 else None
        if checkpoint_dtype is None:
            print(f"  NOTE: checkpoint mixes dtypes {sorted(dtypes)}")
        bootstrap_assets.build_opuslm_assets(
            Path(output_dir),
            model=args.model,
            offline_dir=args.assets_from,
            legacy_overrides=not args.no_legacy_overrides,
            checkpoint_dtype=checkpoint_dtype,
        )
        bootstrap_assets.validate(Path(output_dir), args.model)

    print()
    max_shard_bytes = parse_size(args.max_shard_size)
    save_sharded_safetensors(mapped, output_dir, max_shard_bytes)

    print(f"\nDone! vLLM checkpoint saved to: {output_dir}")
    print(f"Files: {sorted(os.listdir(output_dir))}")
    model = args.model or json.loads(
        (Path(output_dir) / "config.json").read_text()
    ).get("model_type", "opuslm")
    suffix = "_dialogue" if model == "opuslm_dialogue" else ""
    print("\nServe it with:")
    print(f"  MODEL_PATH={output_dir} bash examples/espnet/serve_opuslm{suffix}.sh")
    print("Then send a request:")
    print(f"  python examples/espnet/clients/client_opuslm{suffix}.py")


if __name__ == "__main__":
    main()
