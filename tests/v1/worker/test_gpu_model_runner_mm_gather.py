# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for multimodal encoder batching and gathering (model runner V1).

Mirrors tests/v1/worker/test_encoder_runner.py (the V2 runner): the EAGLE/MTP
drafter reads one position ahead of the target (shift_computed_tokens=1). The
+1 look-ahead feature past the processed boundary is used when its encoder
output is present and tolerated (token-embedding fallback) when it is not,
while a miss within the processed range still fails loudly.

`_gather_mm_embeddings` only uses CPU-side state, so it is exercised against a
lightweight stub for `self` instead of a full (CUDA-only) runner.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.attention.cross_attention import _get_cross_slot_mapping
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

HIDDEN = 4


def _feature(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def _gather(features, cached, *, num_scheduled, shift, num_computed=0):
    encoder_cache = {
        f.identifier: torch.arange(
            f.mm_position.length * HIDDEN, dtype=torch.float32
        ).reshape(f.mm_position.length, HIDDEN)
        for f in cached
    }
    req_state = SimpleNamespace(num_computed_tokens=num_computed, mm_features=features)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req0"]),
        requests={"req0": req_state},
        encoder_cache=encoder_cache,
        _get_encoder_output_from_cache=lambda mm_hash: encoder_cache.get(mm_hash),
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=num_scheduled,
        num_scheduled_tokens={"req0": num_scheduled},
    )
    return GPUModelRunner._gather_mm_embeddings(
        runner, scheduler_output, shift_computed_tokens=shift
    )


def test_draft_shift_uses_boundary_feature_when_cached():
    """The drafter's +1 look-ahead reaches the feature at offset ==
    processed_end; when it is already cached it is used for the look-ahead
    position rather than ignored."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # starts exactly at processed_end
    mm_embeds, is_mm_embed = _gather([f0, f1], [f0, f1], num_scheduled=8, shift=1)

    # f0 covers positions 0..6 (+1 skew); f1's first embed covers position 7.
    assert len(mm_embeds) == 2
    assert bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 8


def test_draft_shift_tolerates_missing_boundary_feature():
    """When the +1 look-ahead feature past the processed boundary is not yet
    encoded, fall back to the token embedding instead of raising."""
    f0 = _feature("h0", offset=0, length=8)
    f1 = _feature("h1", offset=8, length=8)  # boundary feature, not cached
    mm_embeds, is_mm_embed = _gather([f0, f1], [f0], num_scheduled=8, shift=1)

    assert len(mm_embeds) == 1  # only f0; f1's boundary position falls back
    assert not bool(is_mm_embed[7])
    assert int(is_mm_embed.sum()) == 7


def test_draft_shift_raises_on_interior_miss():
    """A miss for a feature within the processed range (not the look-ahead
    boundary) is a real invariant violation, even on the drafter path."""
    f0 = _feature("h0", offset=0, length=8)  # interior, within processed range
    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather([f0], [], num_scheduled=8, shift=1)


def test_target_path_raises_on_encoder_cache_miss():
    """On the target path (no shift) a miss is a real invariant violation."""
    f0 = _feature("h0", offset=0, length=8)
    with pytest.raises(RuntimeError, match="Encoder cache miss"):
        _gather([f0], [], num_scheduled=8, shift=0)


def _encoder_feature(identifier: str, value: float, length: int):
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem(
            encoder_states=MultiModalFieldElem(
                data=torch.full((length, HIDDEN), value, dtype=torch.float32),
                field=MultiModalBatchedField(keep_on_cpu=True),
            )
        ),
        modality="audio",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=length),
    )


def _encoder_runner(req_ids: list[str], *, is_encoder_decoder: bool = True):
    features = {
        "new_a": _encoder_feature("audio_a", 11, 2),
        "new_b": _encoder_feature("audio_b", 22, 3),
    }
    encoder_cache = {}
    runner = SimpleNamespace(
        model_config=SimpleNamespace(is_encoder_decoder=is_encoder_decoder),
        input_batch=SimpleNamespace(req_ids=req_ids),
        requests={
            req_id: SimpleNamespace(mm_features=[feature])
            for req_id, feature in features.items()
        },
        model=SimpleNamespace(
            embed_multimodal=lambda encoder_states: tuple(encoder_states)
        ),
        device=torch.device("cpu"),
        observability_config=None,
        lora_config=None,
        is_multimodal_pruning_enabled=False,
        requires_sequential_video_encoding=False,
        encoder_cudagraph_manager=None,
        timed_encoder_operation=lambda *args: nullcontext(),
        _cache_encoder_output=lambda mm_hash, output, *args: encoder_cache.update(
            {mm_hash: output}
        ),
    )
    # Bind the production method; no copied ordering logic or CUDA runner.
    runner._batch_mm_inputs_from_scheduler = (
        GPUModelRunner._batch_mm_inputs_from_scheduler.__get__(runner)
    )
    return runner, features, encoder_cache


@pytest.mark.parametrize(
    "req_ids,scheduled_req_ids",
    [
        (["new_a", "decode", "new_b"], ["new_b", "new_a"]),
        (["decode", "new_b", "new_a"], ["new_a", "new_b"]),
    ],
)
def test_encoder_decoder_outputs_match_cross_cache_slots(req_ids, scheduled_req_ids):
    """Insertion/backend reordering must not write one audio into another cache."""
    runner, features, encoder_cache = _encoder_runner(req_ids)
    scheduler_output = SimpleNamespace(
        scheduled_encoder_inputs={req_id: [0] for req_id in scheduled_req_ids},
        ec_manager_metadata=None,
        free_encoder_mm_hashes=[],
    )
    scheduled_order = list(scheduler_output.scheduled_encoder_inputs)
    outputs = GPUModelRunner._execute_mm_encoder(runner, scheduler_output)

    block_size = 4
    block_table = torch.tensor([[3], [5], [7]], dtype=torch.int32)
    encoder_seq_lens = np.array(
        [
            features[req_id].mm_position.length if req_id in features else 0
            for req_id in req_ids
        ],
        dtype=np.int32,
    )
    # CrossAttentionBuilder masks the existing decode row to zero before this
    # production slot builder. Scatter the actual encoder output as the kernel
    # does, so an incorrect order corrupts the request-specific memory.
    slot_mapping = _get_cross_slot_mapping(
        encoder_seq_lens,
        block_table,
        SimpleNamespace(block_size=block_size),
        torch.device("cpu"),
    )
    cache = torch.full((32, HIDDEN), -1.0)
    cache[slot_mapping] = torch.cat(outputs)
    for req_index, req_id in enumerate(req_ids):
        block_start = int(block_table[req_index, 0]) * block_size
        if req_id == "decode":
            assert torch.all(cache[block_start : block_start + block_size] == -1)
        else:
            feature = features[req_id]
            expected = feature.data["encoder_states"].data
            torch.testing.assert_close(
                cache[block_start : block_start + feature.mm_position.length], expected
            )
            torch.testing.assert_close(encoder_cache[feature.identifier], expected)
    assert list(scheduler_output.scheduled_encoder_inputs) == scheduled_order


def test_decoder_only_encoder_batch_keeps_scheduler_order():
    runner, _, _ = _encoder_runner(
        ["new_a", "decode", "new_b"], is_encoder_decoder=False
    )
    scheduler_output = SimpleNamespace(
        scheduled_encoder_inputs={"new_b": [0], "new_a": [0]}
    )
    hashes, _, lora_refs = runner._batch_mm_inputs_from_scheduler(scheduler_output)
    assert hashes == ["audio_b", "audio_a"]
    assert [req_id for req_id, _ in lora_refs] == ["new_b", "new_a"]


def test_encoder_decoder_outputs_follow_actual_backend_reorder():
    # Two newly admitted prefills fill holes before a surviving decode request.
    req_ids = ["new_a", "new_b", "decode"]

    def swap_states(left, right):
        req_ids[left], req_ids[right] = req_ids[right], req_ids[left]

    input_batch = SimpleNamespace(
        req_ids=req_ids,
        num_computed_tokens_cpu=np.array([0, 0, 4]),
        num_prompt_tokens=np.array([4, 4, 4]),
        swap_states=swap_states,
    )
    assert reorder_batch_to_split_decodes_and_prefills(
        input_batch,
        SimpleNamespace(num_scheduled_tokens={"new_a": 4, "new_b": 4, "decode": 1}),
    )
    # The production reorder preserves new_b's already-correct region and swaps
    # new_a with decode, reversing the two encoder requests.
    assert req_ids == ["decode", "new_b", "new_a"]
    test_encoder_decoder_outputs_match_cross_cache_slots(req_ids, ["new_a", "new_b"])


def test_encoder_batch_without_scheduled_inputs_is_empty():
    runner, _, _ = _encoder_runner(["decode"])
    assert runner._batch_mm_inputs_from_scheduler(
        SimpleNamespace(scheduled_encoder_inputs={})
    ) == ([], [], [])


def test_multistream_trace_rejects_custom_engine_logits_processors():
    """A later custom processor invalidates the model-level probability trace."""
    runner = SimpleNamespace(
        model_config=SimpleNamespace(logits_processors=["custom.Processor"]),
        input_batch=SimpleNamespace(
            num_reqs=1,
            req_ids=["request"],
            num_computed_tokens_cpu=np.array([0]),
        ),
        requests={
            "request": SimpleNamespace(
                num_tokens=4,
                sampling_params=SimpleNamespace(
                    extra_args={"collect_token_trace": True}
                ),
            )
        },
    )
    with pytest.raises(ValueError, match="custom engine logits processors"):
        GPUModelRunner._sync_audio_batch_state(
            runner,
            SimpleNamespace(_per_req_config={}),
            SimpleNamespace(num_scheduled_tokens={"request": 4}),
        )
