# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os
from collections.abc import Sequence

from vllm.logger import init_logger
from vllm.sampling_params import RepetitionDetectionParams
from vllm.transformers_utils.configs.opuslm import AUDIO_OUT_MODES, MODE_TO_TASK_ALIAS
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# Traces the OpusLM EOS-deferral decision. Shares the switch with the audio
# phase tracer in the model runner: a request that returns no audio is almost
# always a disagreement between these two, so they have to be readable together.
ESPNET_AUDIO_DEBUG = os.environ.get("VLLM_ESPNET_AUDIO_DEBUG", "0") == "1"


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


# OpusLM task tokens whose target segment is codec frames, used only as a
# fallback when the caller passes `opuslm_delay_steps` without an explicit
# task-id set. The scheduler always derives the real ids from the model config.
_DEFAULT_OPUSLM_TTS_TASK_IDS = frozenset({81, 82, 89})

# `extra_args["task"]` values whose responses contain an audio stream. The
# text-output tasks (asr, textlm, text_dialogue) are deliberately absent: they
# never enter the audio phase, so they have no flush tail to wait for.
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
    }
)


def _is_opuslm_tts_request(
    request: Request,
    opuslm_tts_task_ids: frozenset[int] | set[int] | None,
) -> bool:
    """Whether this request generates an audio stream (and therefore emits the
    ARDelay flush tail that the EOS deferral is about)."""
    extra_args = (
        request.sampling_params.extra_args
        if request.sampling_params is not None
        else None
    )
    if extra_args:
        # `mode` is what clients actually send, and it is the same predicate the
        # model runner uses to decide whether to collect audio, so keying on it
        # keeps the stop rule and the audio egress from disagreeing.
        mode = extra_args.get("mode")
        if isinstance(mode, str):
            mode_norm = mode.strip().lower()
            if mode_norm in AUDIO_OUT_MODES:
                return True
            if mode_norm in MODE_TO_TASK_ALIAS:
                # A known text-output mode: no audio stream, no flush tail.
                return False
        task = extra_args.get("task")
        if isinstance(task, str):
            return task.strip().lower() in _OPUSLM_AUDIO_TASKS
        if isinstance(task, int) and not isinstance(task, bool):
            return int(task) in (
                opuslm_tts_task_ids
                if opuslm_tts_task_ids is not None
                else _DEFAULT_OPUSLM_TTS_TASK_IDS
            )

    # No per-request hint: fall back to the task token, which is the second
    # prompt token of every laid-out opuslm prompt. This is also the only signal
    # a CFG shadow reaching here without extra_args would have.
    tts_task_ids = (
        opuslm_tts_task_ids
        if opuslm_tts_task_ids is not None
        else _DEFAULT_OPUSLM_TTS_TASK_IDS
    )
    prompt_token_ids = request.prompt_token_ids
    return (
        prompt_token_ids is not None
        and len(prompt_token_ids) >= 2
        and int(prompt_token_ids[1]) in tts_task_ids
    )


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
        if ESPNET_AUDIO_DEBUG:
            prompt_token_ids = request.prompt_token_ids
            logger.info(
                "[espnet-stop] req=%s n_out=%d eos=%s prompt_head=%s "
                "delay_steps=%s task_ids=%s extra_args=%s is_tts=%s defer=%s",
                request.request_id,
                request.num_output_tokens,
                sampling_params.eos_token_id,
                list(prompt_token_ids[:3]) if prompt_token_ids else None,
                opuslm_delay_steps,
                sorted(opuslm_tts_task_ids) if opuslm_tts_task_ids else None,
                sampling_params.extra_args,
                _is_opuslm_tts_request(request, opuslm_tts_task_ids),
                opuslm_delay_steps is not None
                and _should_defer_opuslm_eos_stop(
                    request, opuslm_delay_steps, opuslm_tts_task_ids
                ),
            )
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
