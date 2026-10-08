# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OWSM vocabulary fidelity and encoder/decoder prompt separation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import sentencepiece as spm
import torch

from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.owsm_processing import (
    OWSMDummyInputsBuilder,
    OWSMMultiModalProcessor,
    OWSMProcessingInfo,
    owsm_encoder_length,
)
from vllm.multimodal.processing.context import InputProcessingContext, TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.tokenizers.owsm import OWSMTokenizer
from vllm.transformers_utils.configs.owsm import OWSMConfig


@pytest.fixture
def tokenizer_path(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nwe test speech recognition\nhello speech\n" * 20)
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "bpe"),
        vocab_size=25,
        hard_vocab_limit=False,
        bos_id=-1,
        eos_id=-1,
    )
    pieces = spm.SentencePieceProcessor(model_file=str(tmp_path / "bpe.model"))
    # Reordering deliberately invalidates the assumption SP IDs equal ESPnet IDs.
    tokens = ["<blank>", "<sos>", "<eos>", "<eng>", "<asr>", "<notimestamps>"]
    tokens += [pieces.id_to_piece(i) for i in reversed(range(pieces.vocab_size()))]
    (tmp_path / "tokens.txt").write_text("\n".join(tokens) + "\n")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            }
        )
    )
    return tmp_path


def test_tokenizer_preserves_checkpoint_ids_and_special_prefix(tokenizer_path):
    tokenizer = OWSMTokenizer.from_pretrained(tokenizer_path)
    prefix = "<sos><eng><asr><notimestamps>"
    ids = tokenizer.encode(prefix + "hello world")
    assert ids[:4] == [1, 3, 4, 5]
    assert tokenizer.decode(ids, skip_special_tokens=True) == "hello world"
    assert (
        tokenizer.encode("hello")[0]
        == tokenizer.get_vocab()[tokenizer.sp.encode("hello", out_type=str)[0]]
    )
    assert tokenizer.encode(prefix, add_special_tokens=True) == [1, 3, 4, 5]


@pytest.mark.parametrize("input_layer,expected", [("conv2d8", 374), ("conv2d", 749)])
def test_encoder_length_accounts_for_all_subsampling_convolutions(
    input_layer, expected
):
    config = OWSMConfig(
        espnet_config={
            "frontend_conf": {"n_fft": 512, "hop_length": 160},
            "encoder_conf": {"input_layer": input_layer},
        }
    )
    assert owsm_encoder_length(480000, config) == expected
    with pytest.raises(ValueError, match="short"):
        owsm_encoder_length(1, config)


def make_processor(path, pad_to_training_window):
    tokenizer = OWSMTokenizer.from_pretrained(path)
    config = OWSMConfig(
        espnet_config={
            "frontend_conf": {"n_fft": 512, "hop_length": 160},
            "encoder_conf": {"input_layer": "conv2d8"},
            "preprocessor_conf": {"speech_length": 30},
            "token_list": tokenizer.tokens,
        },
        vocab_size=len(tokenizer),
        pad_to_training_window=pad_to_training_window,
    )
    mm_config = MultiModalConfig(limit_per_prompt={"audio": 1})
    model = SimpleNamespace(
        model=str(path),
        hf_config=config,
        dtype=torch.float32,
        max_model_len=2048,
        encoder_config=None,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
    )
    info = OWSMProcessingInfo(InputProcessingContext(model, tokenizer))
    return OWSMMultiModalProcessor(info, OWSMDummyInputsBuilder(info))


@pytest.mark.parametrize("pad,expected", [(True, 374), (False, 11)])
def test_processor_keeps_decoder_prefix_and_reserves_exact_encoder_memory(
    tokenizer_path,
    pad,
    expected,
):
    processor = make_processor(tokenizer_path, pad)
    items = processor.info.parse_mm_data(
        {
            "audio": (np.zeros(16000, dtype=np.float32), 16000),
        }
    )
    results = []
    for prompt in ("<sos><eng><asr><notimestamps>", [1, 3, 4, 5]):
        result = processor.apply(ProcessorInputs(prompt, items), TimingContext(False))
        assert result["prompt_token_ids"] == [1, 3, 4, 5]
        assert len(result["encoder_prompt_token_ids"]) == expected
        assert result["mm_placeholders"]["audio"][0].length == expected
        results.append(result["mm_kwargs"]["audio"][0].get_data())
    assert torch.equal(results[0]["speech"], results[1]["speech"])
    assert int(results[0]["speech_lengths"]) == (480000 if pad else 16000)


def test_processor_rejects_audio_beyond_frozen_window(tokenizer_path):
    processor = make_processor(tokenizer_path, True)
    with pytest.raises(ValueError, match="exceeds"):
        processor._call_hf_processor("", {"audios": [np.zeros(480001)]}, {}, {})
