# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace already sampled actions, with no GPU model required."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.espnet_token_trace import (
    begin_token_trace_step,
    get_espnet_token_traces,
    observe_primary_logits,
    observe_secondary_samples,
    record_sampled_token_step,
)


def test_trace_preserves_joint_samples_and_excludes_forced_delay_padding():
    request = "1_external-abcdef12"
    model = SimpleNamespace(
        config=SimpleNamespace(nq=3),
        _current_batch_req_ids=[request],
        _per_req_config={request: {"collect_token_trace": True}},
    )
    begin_token_trace_step(model)
    primary = torch.tensor([[0.1, 0.2, 0.3]])
    secondary = torch.tensor([[0.5, -torch.inf, 0.9]])
    original = secondary.clone()
    observe_primary_logits(model, primary)
    observe_secondary_samples(
        model, [0], 1, secondary, torch.tensor([2]), torch.tensor([True]), 1.0, 0
    )
    observe_secondary_samples(
        model, [0], 2, secondary, torch.tensor([0]), torch.tensor([False]), 1.0, 0
    )
    params = SimpleNamespace(
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    record_sampled_token_step(model, [request], [1], [params])
    result = get_espnet_token_traces(model, ["external"], pop=True)["external:1"]
    assert result["token_ids"] == [[1, 2, 0]]
    assert result["stream_mask"] == [[True, True, False]]
    expected = primary.log_softmax(-1)[0, 1] + secondary.log_softmax(-1)[0, 2]
    assert result["log_probs"][0] == pytest.approx(float(expected))
    assert result["stream_log_probs"][0] == pytest.approx(
        [
            float(primary.log_softmax(-1)[0, 1]),
            float(secondary.log_softmax(-1)[0, 2]),
            0.0,
        ]
    )
    assert sum(result["stream_log_probs"][0]) == result["log_probs"][0]
    assert result["sampling_contracts"][0][1]["allowed_ranges"] == [[0, 1], [2, 3]]
    torch.testing.assert_close(secondary, original)
    assert not get_espnet_token_traces(model, ["external"])


def test_unrequested_trace_does_not_retain_model_logits():
    model = SimpleNamespace(
        config=SimpleNamespace(num_stream=2),
        _current_batch_req_ids=["request"],
        _per_req_config={},
    )
    begin_token_trace_step(model)
    observe_primary_logits(model, torch.ones(1, 100))
    assert not model._espnet_token_trace_pending
    assert not model._espnet_token_traces


def test_dummy_profile_codec_samples_have_no_request_trace():
    model = SimpleNamespace(
        config=SimpleNamespace(num_stream=2),
        _current_batch_req_ids=[],
        _per_req_config={},
    )
    begin_token_trace_step(model)
    logits = torch.tensor([[0.5, 0.9]])
    sampled = torch.tensor([1])
    observe_secondary_samples(
        model, [0], 1, logits, sampled, torch.tensor([True]), 1.0, 0
    )
    assert not model._espnet_token_trace_pending
    assert not model._espnet_token_traces
    torch.testing.assert_close(logits, torch.tensor([[0.5, 0.9]]))
    torch.testing.assert_close(sampled, torch.tensor([1]))


def test_forced_primary_token_is_not_a_policy_action():
    model = SimpleNamespace(
        config=SimpleNamespace(num_stream=1),
        _current_batch_req_ids=["request"],
        _per_req_config={"request": {"collect_token_trace": True}},
    )
    begin_token_trace_step(model)
    observe_primary_logits(model, torch.tensor([[-torch.inf, 0.0]]))
    params = SimpleNamespace(
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    record_sampled_token_step(model, ["request"], [1], [params])
    trace = get_espnet_token_traces(model, ["request"])["request:0"]
    assert trace["action_mask"] == [False] and trace["log_probs"] == [0.0]
    assert trace["stream_log_probs"] == [[0.0]]


@pytest.mark.parametrize(
    "internal_id,index",
    [("1_task-abcdef12", 0), ("2_1_task-abcdef12", 2), ("1_task", 0)],
)
def test_trace_keeps_numeric_external_request_ids(internal_id, index):
    model = SimpleNamespace(_espnet_token_traces={internal_id: {"token_ids": [[7]]}})
    assert get_espnet_token_traces(model, ["1_task"]) == {
        f"1_task:{index}": {"token_ids": [[7]]}
    }


@pytest.mark.parametrize("request_ids", [["task", "1_task"], ["1_task", "task"]])
def test_trace_exact_external_match_precedes_child_parsing(request_ids):
    model = SimpleNamespace(
        _espnet_token_traces={"1_task-abcdef12": {"token_ids": [[9]]}}
    )
    assert get_espnet_token_traces(model, request_ids) == {
        "1_task:0": {"token_ids": [[9]]}
    }


@pytest.mark.parametrize(
    "transform",
    [
        {"allowed_token_ids": [0]},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["word"]},
        {"min_tokens": 1},
        {"structured_outputs": object()},
        {"thinking_token_budget": 1},
    ],
)
def test_trace_rejects_unobserved_standard_sampler_transforms(transform):
    """Model-level observations cannot score a later modified distribution."""
    model = SimpleNamespace(
        config=SimpleNamespace(num_stream=1),
        _current_batch_req_ids=["request"],
        _per_req_config={"request": {"collect_token_trace": True}},
    )
    begin_token_trace_step(model)
    observe_primary_logits(model, torch.tensor([[0.0, 0.0]]))
    params = SimpleNamespace(
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        **transform,
    )
    with pytest.raises(ValueError, match="unmodified standard sampler"):
        record_sampled_token_step(model, ["request"], [0], [params])


@pytest.mark.parametrize("top_k", [1, 2])
def test_trace_rejects_primary_top_k_with_sampler_dependent_ties(top_k):
    """A sampler may retain more than k equal logits at the cutoff."""
    model = SimpleNamespace(
        config=SimpleNamespace(num_stream=1),
        _current_batch_req_ids=["request"],
        _per_req_config={"request": {"collect_token_trace": True}},
    )
    begin_token_trace_step(model)
    observe_primary_logits(model, torch.zeros(1, 3))
    params = SimpleNamespace(
        temperature=1.0,
        top_k=top_k,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
    with pytest.raises(ValueError, match="full primary sampling support"):
        record_sampled_token_step(model, ["request"], [0], [params])
