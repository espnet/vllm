# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in traces of ESPnet's real primary and internal codec samples.

The trace records decisions the existing sampler has already made. It never
samples, changes logits, or edits a request. ``LLM.apply_model`` retrieves these
small CPU records after generation, including all parallel codec streams.
"""

import math
from collections.abc import Sequence
from typing import Any

import regex as re
import torch


def _enabled(model, request_id):
    config = model._per_req_config.get(request_id, {})
    return bool(config.get("collect_token_trace")) and not config.get("is_shadow")


def begin_token_trace_step(model):
    """Reset pending observations for the current batch, retaining finished traces."""
    if not hasattr(model, "_espnet_token_trace_pending"):
        model._espnet_token_trace_pending = {}
        model._espnet_token_traces = {}
    for request_id in model._current_batch_req_ids:
        if _enabled(model, request_id):
            model._espnet_token_trace_pending[request_id] = {}


def _ranges(mask):
    # Run-length encode categorical support instead of serializing 160k bits.
    padded = torch.nn.functional.pad(mask.to(torch.int8), (1, 1))
    changes = (padded[1:] - padded[:-1]).nonzero().flatten().cpu().tolist()
    return [[int(lo), int(hi)] for lo, hi in zip(changes[::2], changes[1::2])]


def _contract(logits, temperature, top_k):
    return {
        "allowed_ranges": _ranges(torch.isfinite(logits)),
        "temperature": float(temperature),
        "top_k": int(top_k),
    }


def observe_primary_logits(model, logits):
    """Retain logits before the standard sampler; no tensor is modified."""
    pending = getattr(model, "_espnet_token_trace_pending", {})
    for index, request_id in enumerate(model._current_batch_req_ids):
        if index < logits.shape[0] and _enabled(model, request_id):
            pending.setdefault(request_id, {})[0] = {
                "logits": logits[index].detach().float().clone()
            }


def observe_secondary_samples(
    model, positions, stream, logits, sampled, action_mask, temperature, top_k
):
    """Record the tokens just produced by an internal codec sampler.

    ``action_mask`` is false for forced delay warmup/flush padding. Sampling
    these positions internally does not turn the subsequently forced value
    into a policy action.
    """
    pending = getattr(model, "_espnet_token_trace_pending", {})
    for row, position in enumerate(positions):
        request_id = model._current_batch_req_ids[position]
        if not _enabled(model, request_id):
            continue
        active = bool(action_mask[row])
        value = int(sampled[row])
        score = 0.0
        if active:
            if temperature <= 0:
                raise ValueError(
                    "ESPnet RL token tracing requires stochastic codec sampling"
                )
            distribution = logits[row].detach().float() / temperature
            if 0 < top_k < distribution.shape[-1]:
                top_values, top_ids = distribution.topk(top_k)
                # Same top-k candidate set used by _top_k_sample.
                matches = (top_ids == value).nonzero().flatten()
                if len(matches) != 1:
                    raise RuntimeError(
                        "Sampled codec token is outside its recorded top-k support"
                    )
                score = float(top_values.log_softmax(-1)[matches[0]])
            else:
                score = float(distribution.log_softmax(-1)[value])
        pending.setdefault(request_id, {})[stream] = {
            "token_id": value,
            "active": active,
            "log_prob": score,
            "contract": _contract(
                logits[row], max(temperature, 1.0) if not active else temperature, top_k
            ),
        }


def record_sampled_token_step(model, request_ids, sampled_tokens, sampling_params):
    """Join the standard sampler's actual output with internal stream samples."""
    pending = getattr(model, "_espnet_token_trace_pending", {})
    traces = getattr(model, "_espnet_token_traces", {})
    count = int(getattr(model.config, "num_stream", getattr(model.config, "nq", 1)))
    for request_id, token, params in zip(
        request_ids, sampled_tokens, sampling_params, strict=True
    ):
        if token is None or not _enabled(model, request_id):
            continue
        observations = pending.get(request_id, {})
        primary = observations.get(0)
        if primary is None:
            raise RuntimeError(
                f"Missing primary logits for traced ESPnet request {request_id}"
            )
        if params is None or params.temperature <= 0:
            raise ValueError("ESPnet RL trace requires stochastic standard sampling")
        if (
            params.top_p != 1.0
            or params.min_p != 0.0
            or params.repetition_penalty != 1.0
            or params.presence_penalty != 0.0
            or params.frequency_penalty != 0.0
        ):
            raise ValueError(
                "ESPnet RL trace requires top_p=1, min_p=0 "
                "and disabled repetition penalties"
            )
        logits = primary["logits"]
        active = int(torch.isfinite(logits).sum()) > 1
        temperature = float(params.temperature)
        top_k = int(params.top_k)
        distribution = logits / temperature
        if 0 < top_k < distribution.shape[-1]:
            values, indices = distribution.topk(top_k)
            matches = (indices == token).nonzero().flatten()
            if len(matches) != 1:
                raise RuntimeError(
                    "Primary sampled token is outside the recorded top-k support"
                )
            score = float(values.log_softmax(-1)[matches[0]])
        else:
            score = float(distribution.log_softmax(-1)[token])
        tokens, masks = [int(token)], [active]
        scores = [score if active else 0.0]
        contracts = [_contract(logits, temperature, top_k)]
        for stream in range(1, count):
            observation = observations.get(stream)
            if observation is None:
                tokens.append(0)
                masks.append(False)
                scores.append(0.0)
                contracts.append(
                    {"allowed_ranges": [[0, 1]], "temperature": 1.0, "top_k": 0}
                )
            else:
                tokens.append(observation["token_id"])
                masks.append(observation["active"])
                scores.append(observation["log_prob"])
                contracts.append(observation["contract"])
        joint = sum(scores)
        if not math.isfinite(joint):
            raise RuntimeError("Nonfinite ESPnet sampled action log probability")
        trace = traces.setdefault(
            request_id,
            {
                "token_ids": [],
                "log_probs": [],
                "action_mask": [],
                "stream_mask": [],
                "sampling_contracts": [],
                "codec_frame_mask": [],
            },
        )
        trace["token_ids"].append(tokens)
        trace["log_probs"].append(joint)
        trace["action_mask"].append(any(masks))
        trace["stream_mask"].append(masks)
        trace["sampling_contracts"].append(contracts)
        trace["codec_frame_mask"].append(any(stream > 0 for stream in observations))
        pending.pop(request_id, None)  # Release retained vocabulary logits.


def get_espnet_token_traces(
    model, request_ids: Sequence[str], pop=True
) -> dict[str, Any]:
    """Public ``LLM.apply_model`` callback, keys are external-request-id:index.

    vLLM internally appends a UUID fragment and names an n-sample child
    ``index_parent``. Strip only that documented engine suffix; exact request
    IDs are checked first. Completed traces remain until explicitly collected.
    """
    model = getattr(model, "runnable", model)
    traces = getattr(model, "_espnet_token_traces", {})
    result = {}

    def matches(parent, external):
        return parent == external or re.fullmatch(
            re.escape(external) + r"-[0-9a-f]{8}", parent
        )

    for internal_id, trace in list(traces.items()):
        candidates = [
            (external, 0) for external in request_ids if matches(internal_id, external)
        ]
        if not candidates:
            child = re.fullmatch(r"(\d+)_(.+)", internal_id)
            if child:
                candidates = [
                    (external, int(child[1]))
                    for external in request_ids
                    if matches(child[2], external)
                ]
        if len(candidates) > 1:
            raise RuntimeError(f"Ambiguous ESPnet external identity {internal_id}")
        if candidates:
            external, index = candidates[0]
            key = f"{external}:{index}"
            if key in result:
                raise RuntimeError(f"Ambiguous ESPnet trace identity {key}")
            result[key] = trace
            if pop:
                traces.pop(internal_id)
    return result
