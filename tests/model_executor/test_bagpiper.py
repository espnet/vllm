# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.bagpiper import (
    BagpiperForConditionalGeneration,
)


def _make_bagpiper_stub() -> BagpiperForConditionalGeneration:
    model = object.__new__(BagpiperForConditionalGeneration)
    model.config = SimpleNamespace(
        audio_temperature=0.8,
        audio_topk=20,
        vocab_size=100,
    )
    model._current_batch_req_ids = ["r0", "r1", "r2", "r3", "r4"]
    model._per_req_config = {}
    model._audio_batch_layout = {}
    return model


@pytest.mark.parametrize("chunks", [[(0, 6)], [(0, 3), (3, 2), (5, 1)]])
def test_stream_embeddings_replay_positions_in_mixed_batches(chunks):
    """Prefill and single-token replay recover the same streams after reorder."""
    model = _make_bagpiper_stub()
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=lambda ids: ids.float().unsqueeze(-1))
    )
    model._stream_tokens_by_position = {
        "audio": {
            3: torch.tensor([1, 0]),
            4: torch.tensor([2, 3]),
            5: torch.tensor([4, 5]),
        },
        "decode": {8: torch.tensor([6, 7])},
    }
    # The most recent buffer must not leak into historical/prompt positions.
    model._stream_buffer_dict = {"audio": torch.tensor([99, 99])}
    replayed = []
    for start, count in chunks:
        model._current_batch_req_ids = ["decode", "audio"]
        model._audio_batch_layout = {"decode": (8, 1, 9), "audio": (start, count, 6)}
        ids = torch.full((count + 1,), 10)
        result = model._apply_stream_embeddings(
            ids, ids.float().unsqueeze(-1), torch.ones_like(ids, dtype=torch.bool)
        )
        assert result[0].item() == 23
        replayed.extend(result[1:, 0].tolist())
    assert replayed == [10, 10, 10, 11, 15, 19]


def test_partial_replay_does_not_sample_or_advance_audio_phase():
    model = _make_bagpiper_stub()
    model._current_batch_req_ids = ["audio"]
    model._audio_batch_layout = {"audio": (3, 1, 6)}
    model._per_req_config = {"audio": {"mode": "text_audio", "phase": "audio"}}
    model.config.eot_token_id = 4
    assert not model._can_sample_audio(0)
    model._update_text_audio_phase(torch.tensor([4]))
    assert model._per_req_config["audio"]["phase"] == "audio"
    model._audio_batch_layout["audio"] = (5, 1, 6)
    assert model._can_sample_audio(0)


def test_main_and_shadow_cache_use_their_own_positions_and_are_cleaned_up():
    model = _make_bagpiper_stub()
    model._audio_batch_layout = {"main": (100, 1, 101), "shadow": (20, 1, 21)}
    model._stream_tokens_by_position = {}
    streams = torch.tensor([7, 8])
    model._stream_buffer_dict = {"main": streams, "shadow": streams.clone()}
    model._stream17_history = {}
    model._stream0_history = {}
    for req_id, position in [("main", 101), ("shadow", 21)]:
        model._cache_stream_tokens(req_id)
        assert torch.equal(model._stream_tokens_by_position[req_id][position], streams)
        model.cleanup_request(req_id)
        assert req_id not in model._stream_tokens_by_position


def test_audio_sampling_groups_prioritize_request_xargs():
    model = _make_bagpiper_stub()
    model._per_req_config = {
        "r0": {"audio_temperature": 0.2, "audio_topk": 5},
        "r1": {"audio_temperature": 1.1},
        "r2": {"audio_topk": 999},
        "r3": {"audio_topk": -1},
    }

    groups = model._get_audio_sampling_groups([0, 1, 2, 3, 4], torch.device("cpu"))
    grouped_rows = {(temp, top_k): rows.tolist() for temp, top_k, rows in groups}

    assert grouped_rows[(0.2, 5)] == [0]
    assert grouped_rows[(1.1, 20)] == [1]
    # 999 is clamped to vocab_size=100, and -1 means "full vocab".
    assert grouped_rows[(0.8, 100)] == [2, 3]
    assert grouped_rows[(0.8, 20)] == [4]


def test_audio_sampling_param_normalization():
    assert (
        BagpiperForConditionalGeneration._normalize_audio_temperature("0.5", 0.8) == 0.5
    )
    assert (
        BagpiperForConditionalGeneration._normalize_audio_temperature("bad", 0.8) == 0.8
    )
    assert (
        BagpiperForConditionalGeneration._normalize_audio_temperature(-0.1, 0.8) == 0.8
    )

    assert BagpiperForConditionalGeneration._normalize_audio_top_k("7", 20, 100) == 7
    assert (
        BagpiperForConditionalGeneration._normalize_audio_top_k("999", 20, 100) == 100
    )
    assert BagpiperForConditionalGeneration._normalize_audio_top_k("-1", 20, 100) == 100
    assert BagpiperForConditionalGeneration._normalize_audio_top_k("bad", 20, 100) == 20
