# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpusLM stops on the *second* EOS of an audio segment, not the first.

OpusLM's audio head is delayed by `nq - 1` steps, so an audio segment ends as
`..., EOS, 0 * (nq - 1), EOS`. The first EOS only announces the flush, and the
`nq - 1` codec frames that follow it are the tail of the waveform. Stopping
there truncates every generated clip by that tail and, because the truncation is
silent, the clip still plays.

Nothing downstream re-checks this, so the rule is tested here for all three ways
a request can be recognised as an audio-output one.
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

EOS = 5
DELAY_STEPS = 8  # nq - 1 for the released checkpoints
TTS_TASK_IDS = frozenset({81, 82, 89})

# Plain TTS: <sos/eos>, plain_tts_task, <text_bpe_start/end>, ...
TTS_PROMPT = [5, 82, 35, 20000]
# ASR: <sos/eos>, asr_task, <codec_ssl_start/end>, ... -- answers with text.
ASR_PROMPT = [5, 80, 34, 6000]


def _make_request(prompt_token_ids: list[int], extra_args: dict | None) -> Request:
    params = SamplingParams(max_tokens=4096, extra_args=extra_args)
    params.update_from_generation_config({}, eos_token_id=EOS)
    return Request(
        request_id="test",
        prompt_token_ids=prompt_token_ids,
        sampling_params=params,
        pooling_params=None,
    )


def _stops_at(request: Request) -> bool:
    return check_stop(
        request,
        max_model_len=8192,
        opuslm_delay_steps=DELAY_STEPS,
        opuslm_tts_task_ids=TTS_TASK_IDS,
    )


@pytest.mark.parametrize(
    "prompt,extra_args",
    [
        # What the documented client contract actually sends.
        (TTS_PROMPT, {"mode": "text_audio"}),
        # A task name instead of a mode.
        (TTS_PROMPT, {"task": "tts"}),
        # A raw task token id.
        (TTS_PROMPT, {"task": 81}),
        # Audio dialogue: a different task, same flush tail.
        ([5, 89, 9], {"mode": "audio_dialogue"}),
        # No per-request hint at all. This is the only signal a CFG shadow
        # request has, since it is created without extra_args.
        (TTS_PROMPT, None),
    ],
)
def test_first_eos_defers_and_second_eos_stops(prompt, extra_args):
    request = _make_request(prompt, extra_args)

    # A stretch of codec frames, then the flush-announcing EOS.
    request.append_output_token_ids([6000 + i for i in range(20)])
    request.append_output_token_ids([EOS])
    assert not _stops_at(request), "the flush-announcing EOS must not stop"
    assert not RequestStatus.is_finished(request.status)

    # The flush tail: `nq - 1` frames, then the EOS that really ends it.
    for _ in range(DELAY_STEPS):
        request.append_output_token_ids([0])
        assert not _stops_at(request), "no stop before the tail is complete"
    request.append_output_token_ids([EOS])
    assert _stops_at(request)
    assert request.status == RequestStatus.FINISHED_STOPPED


@pytest.mark.parametrize(
    "prompt,extra_args",
    [
        # Text-output modes have no flush tail: ASR and the two text modes.
        (ASR_PROMPT, {"mode": "audio_text"}),
        (TTS_PROMPT, {"mode": "text_text"}),
        ([5, 88, 9], {"mode": "text_dialogue"}),
        (ASR_PROMPT, {"task": "asr"}),
        # No hint, and the task token in the prompt is a text-output one.
        (ASR_PROMPT, None),
    ],
)
def test_text_output_requests_stop_on_the_first_eos(prompt, extra_args):
    request = _make_request(prompt, extra_args)

    request.append_output_token_ids([20000 + i for i in range(20)])
    request.append_output_token_ids([EOS])
    assert _stops_at(request), "a text answer has no flush tail to wait for"
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_a_non_zero_tail_keeps_deferring():
    """The tail has to be `nq - 1` pad-0 tokens, not just `nq - 1` tokens.

    A second EOS arriving early -- with real codec frames still in the window --
    is another segment boundary, not the end of the flush.
    """
    request = _make_request(TTS_PROMPT, {"mode": "text_audio"})

    request.append_output_token_ids([6000 + i for i in range(20)])
    request.append_output_token_ids([EOS])
    assert not _stops_at(request)

    # Four pads, then a frame, then three pads: eight tokens, but not the tail.
    request.append_output_token_ids([0, 0, 0, 0, 6100, 0, 0, 0])
    request.append_output_token_ids([EOS])
    assert not _stops_at(request)


def test_deferral_is_off_when_the_scheduler_passes_no_delay():
    """Every non-OpusLM model must reach the unmodified upstream behaviour.

    The scheduler only passes `opuslm_delay_steps` for an OpusLM model, so with
    it absent the EOS branch has to stop on the first EOS regardless of what the
    prompt or extra_args look like.
    """
    request = _make_request(TTS_PROMPT, {"mode": "text_audio"})
    request.append_output_token_ids([6000, 6001, EOS])

    assert check_stop(request, max_model_len=8192)
    assert request.status == RequestStatus.FINISHED_STOPPED
