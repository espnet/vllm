# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.opuslm import (
    OpusLMForConditionalGeneration,
    OpusLMMultiModalProcessor,
)
from vllm.model_executor.models.opuslm_dialogue import (
    OpusLMDialogueForConditionalGeneration,
)


@pytest.mark.parametrize(
    "model_class",
    [OpusLMForConditionalGeneration, OpusLMDialogueForConditionalGeneration],
)
def test_text_only_forward_keeps_native_pad_stream_embeddings(model_class):
    """With audio disabled, learned pad streams must survive forward fallback."""
    embedding = torch.nn.Embedding(6, 3)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(18).reshape(6, 3))
    calls = []

    def backbone(**kwargs):
        calls.append(kwargs)
        return kwargs["inputs_embeds"]

    model = SimpleNamespace(
        config=SimpleNamespace(nq=9),
        _stream_buffer_dict={},
        _pad_bias=None,
        _embed_text_input_ids=lambda ids, embed_fn, **kwargs: embed_fn(ids),
    )

    # A callable backbone lets the actual production forward/embed methods run
    # without constructing transformer weights, processors or GPU kernels.
    class Backbone:
        embed_tokens = embedding
        embed_input_ids = embedding

        def __call__(self, **kwargs):
            return backbone(**kwargs)

    model.model = Backbone()
    model.embed_input_ids = lambda ids: model_class.embed_input_ids(model, ids)
    ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    actual = model_class.forward(model, ids, positions)
    expected = embedding(ids) + 8 * embedding(torch.tensor([0]))
    torch.testing.assert_close(actual, expected)
    assert calls[-1]["positions"] is positions
    assert not torch.equal(actual, embedding(ids))


@pytest.mark.parametrize(
    "model_class",
    [OpusLMForConditionalGeneration, OpusLMDialogueForConditionalGeneration],
)
def test_forward_reuses_prepared_embeddings_and_skips_pipeline_embedding(model_class):
    def reject_embedding(ids):
        raise AssertionError("Already prepared or pipeline inputs must not embed twice")

    calls = []

    def backbone(**kwargs):
        calls.append(kwargs)
        return kwargs["intermediate_tensors"] or kwargs["inputs_embeds"]

    model = SimpleNamespace(model=backbone, embed_input_ids=reject_embedding)
    ids, positions = torch.tensor([1]), torch.tensor([0])
    prepared = torch.ones(1, 3)
    assert (
        model_class.forward(model, ids, positions, inputs_embeds=prepared) is prepared
    )
    intermediate = {"hidden_states": torch.ones(1, 3)}
    assert (
        model_class.forward(model, ids, positions, intermediate, prepared)
        is intermediate
    )
    assert calls[-1]["inputs_embeds"] is None


def _make_opuslm_stub() -> OpusLMForConditionalGeneration:
    model = object.__new__(OpusLMForConditionalGeneration)
    model.config = SimpleNamespace(
        codec_token_start=5256,
        codec_per_stream_size=1024,
        num_codec_streams=8,
    )
    return model


def test_delay_deinterleave_matches_reference_slicing():
    model = _make_opuslm_stub()
    codes = torch.arange(1 * 12 * 9, dtype=torch.long).view(1, 12, 9)

    aligned = model._delay_deinterleave(codes)
    # T_original = 12 - 9 + 1 = 4
    assert aligned.shape == (1, 4, 9)
    for stream_idx in range(9):
        expected = codes[:, stream_idx : stream_idx + 4, stream_idx]
        assert torch.equal(aligned[:, :, stream_idx], expected)


def test_global_to_dac_codebook_offsets_and_clamps():
    model = _make_opuslm_stub()
    cfg = model.config

    # [B=1, T=2, S=8] global DAC ids. Some values are intentionally out-of-range
    # to verify clamping to [0, 1023].
    dac_tokens = torch.tensor(
        [
            [
                [
                    cfg.codec_token_start + s * cfg.codec_per_stream_size + 7
                    for s in range(cfg.num_codec_streams)
                ],
                [
                    cfg.codec_token_start + s * cfg.codec_per_stream_size + 5000
                    for s in range(cfg.num_codec_streams)
                ],
            ]
        ],
        dtype=torch.long,
    )

    codebook = model._global_to_dac_codebook(dac_tokens)
    assert codebook.shape == dac_tokens.shape
    assert torch.all(codebook[:, 0, :] == 7)
    assert torch.all(codebook[:, 1, :] == 1023)


def test_resolve_task_token_id_defaults_and_overrides():
    # NOTE: resolve_task_token_id builds its alias table eagerly, so the
    # stub config must also provide the dialogue task token ids.
    cfg = SimpleNamespace(
        textlm_task_token_id=64,
        codec_ssl_asr_task_token_id=80,
        codec_ssl_tts_task_token_id=81,
        codec_ssl_plain_tts_task_token_id=82,
        text_dialogue_task_token_id=88,
        audio_dialogue_task_token_id=89,
    )

    assert (
        OpusLMMultiModalProcessor.resolve_task_token_id(
            cfg,
            has_audio_input=True,
        )
        == 80
    )
    assert (
        OpusLMMultiModalProcessor.resolve_task_token_id(
            cfg,
            has_audio_input=False,
        )
        == 82
    )
    assert (
        OpusLMMultiModalProcessor.resolve_task_token_id(
            cfg,
            has_audio_input=False,
            mode="text_text",
        )
        == 64
    )
    assert (
        OpusLMMultiModalProcessor.resolve_task_token_id(
            cfg,
            has_audio_input=False,
            task="tts",
        )
        == 81
    )


def test_resolve_task_token_id_rejects_invalid_mode():
    cfg = SimpleNamespace(
        textlm_task_token_id=64,
        codec_ssl_asr_task_token_id=80,
        codec_ssl_tts_task_token_id=81,
        codec_ssl_plain_tts_task_token_id=82,
        text_dialogue_task_token_id=88,
        audio_dialogue_task_token_id=89,
    )

    with pytest.raises(ValueError):
        OpusLMMultiModalProcessor.resolve_task_token_id(
            cfg,
            has_audio_input=False,
            mode="audio_audio",
        )
