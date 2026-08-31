# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Sequence

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.request import Request, RequestStatus


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


# OpusLM TTS/dialogue task tokens, used only as a fallback when the caller
# passes `opuslm_delay_steps` without an explicit task-id set. The scheduler
# always derives the real ids from the model config.
_DEFAULT_OPUSLM_TTS_TASK_IDS = frozenset({81, 82, 88, 89})

# `extra_args["task"]` values whose responses contain an audio stream.
_OPUSLM_AUDIO_TASKS = frozenset(
    {
        "tts",
        "plain_tts",
        "codec_ssl_tts",
        "codec_ssl_tts_task",
        "codec_ssl_plain_tts",
        "codec_ssl_plain_tts_task",
        "audio_dialogue",
        "audio_dialogue_task",
        "text_dialogue",
        "text_dialogue_task",
    }
)


def _is_opuslm_tts_request(
    request: Request,
    opuslm_tts_task_ids: frozenset[int] | set[int] | None,
) -> bool:
    """Whether this request generates an audio stream (and therefore emits the
    ARDelay flush tail that the EOS deferral is about)."""
    tts_task_ids = (
        opuslm_tts_task_ids
        if opuslm_tts_task_ids is not None
        else _DEFAULT_OPUSLM_TTS_TASK_IDS
    )
    # The task token is the second prompt token for every opuslm prompt. A miss
    # is not conclusive: a CFG shadow has an all-zero prompt, yet it mirrors its
    # main request's audio tokens, so it has the same flush tail and must defer
    # too. Fall through to the sampling params, which the shadow inherits.
    prompt_token_ids = request.prompt_token_ids
    if (
        prompt_token_ids is not None
        and len(prompt_token_ids) >= 2
        and int(prompt_token_ids[1]) in tts_task_ids
    ):
        return True

    extra_args = (
        request.sampling_params.extra_args
        if request.sampling_params is not None
        else None
    )
    if not extra_args:
        return False
    mode = extra_args.get("mode")
    if mode in ("audio_text", "text_text"):
        # Text-only response: no audio stream, no flush tail.
        return False
    task = extra_args.get("task")
    if not isinstance(task, str):
        return False
    return task.lower() in _OPUSLM_AUDIO_TASKS


def _should_defer_opuslm_eos_stop(
    request: Request,
    opuslm_delay_steps: int,
    opuslm_tts_task_ids: frozenset[int] | set[int] | None,
) -> bool:
    """Whether this EOS is the *first* one of an OpusLM ARDelay flush tail.

    OpusLM's audio head is delayed by `nq - 1` steps, so the model emits
    `..., EOS, 0 * (nq - 1), EOS`: the first EOS only announces the flush and
    the codec frames that follow it are still needed. Only the second EOS --
    the one preceded by exactly `nq - 1` pad-0 tokens -- may stop the request.
    """
    if opuslm_delay_steps <= 0:
        return False
    if not _is_opuslm_tts_request(request, opuslm_tts_task_ids):
        return False
    if request.num_output_tokens <= opuslm_delay_steps:
        # Too early for the flush to have completed.
        return True
    # The `opuslm_delay_steps` tokens right before this EOS.
    window = request.output_token_ids[-(opuslm_delay_steps + 1) : -1]
    if len(window) < opuslm_delay_steps:
        return True
    # An all-zero window means the flush is done and this EOS is the real one.
    return any(tok != 0 for tok in window)


def check_stop(
    request: Request,
    max_model_len: int,
    *,
    opuslm_delay_steps: int | None = None,
    opuslm_tts_task_ids: frozenset[int] | set[int] | None = None,
) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        if opuslm_delay_steps is not None and _should_defer_opuslm_eos_stop(
            request, opuslm_delay_steps, opuslm_tts_task_ids
        ):
            # OpusLM: this is the flush-announcing EOS, keep generating.
            return False
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
