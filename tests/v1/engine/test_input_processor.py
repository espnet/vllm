# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import SchedulerConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor


@pytest.mark.parametrize(
    "max_seqs, cfg, error",
    [
        (1, 3.0, "max_num_seqs >= 2"),
        (2, 3.0, None),
        (1, 1.0, None),
        (1, None, None),
    ],
)
def test_cfg_rejects_unschedulable_pair(max_seqs, cfg, error):
    """Reject impossible CFG capacity before a request reaches EngineCore."""
    processor = InputProcessor.__new__(InputProcessor)
    processor.scheduler_config = SchedulerConfig(
        max_num_seqs=max_seqs,
        max_num_batched_tokens=32,
        max_model_len=32,
        is_encoder_decoder=False,
    )
    processor.model_config = SimpleNamespace(return_sampling_mask=False)
    processor.speculative_config = None
    processor.structured_outputs_config = None
    processor.renderer = SimpleNamespace(tokenizer=None)
    params = SamplingParams(extra_args={"cfg": cfg})

    # The sampling-parameter suite covers model/tokenizer validation; this
    # regression concerns admission capacity, which needs no loaded model.
    with patch.object(SamplingParams, "verify"):
        if error:
            with pytest.raises(VLLMValidationError, match=error):
                processor._validate_params(params, ("generate",))
        else:
            processor._validate_params(params, ("generate",))
