# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw waveform processing and exact encoder lengths for OWSM."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.renderers import TokenizeParams
from vllm.tokenizers.owsm import resolve_owsm_language_symbol
from vllm.transformers_utils.configs.owsm import OWSMConfig


def owsm_encoder_length(num_samples: int, config: OWSMConfig) -> int:
    native = config.espnet_config
    if native.get("frontend", "default") != "default":
        raise ValueError(
            "OWSM waveform processing requires the default ESPnet frontend"
        )
    frontend = native.get("frontend_conf", {})
    if not frontend.get("apply_stft", True):
        raise ValueError("OWSM waveform processing requires STFT")
    n_fft = frontend.get("n_fft", 512)
    padding = n_fft // 2 if frontend.get("center", True) else 0
    length = (num_samples + 2 * padding - n_fft) // frontend.get("hop_length", 128) + 1
    layer = native.get("encoder_conf", {}).get("input_layer", "conv2d")
    convolutions = {
        "conv2d": [(3, 2), (3, 2)],
        "conv2d4": [(3, 2), (3, 2)],
        "conv2d8": [(3, 2), (3, 2), (3, 2)],
        "conv2d6": [(3, 2), (5, 3)],
        "conv2d2": [(3, 2), (3, 1)],
        "linear": [],
        "None": [],
        None: [],
    }
    if layer not in convolutions:
        raise ValueError(f"Unsupported OWSM encoder input_layer: {layer}")
    if native.get("preencoder") or native.get("postencoder"):
        raise ValueError(
            "Custom pre/postencoder length transforms need an explicit adapter"
        )
    for kernel, stride in convolutions[layer]:
        length = (length - kernel) // stride + 1
    if length <= 0 or (frontend.get("center", True) and num_samples <= n_fft // 2):
        raise ValueError("Audio is too short for OWSM's STFT and subsampling")
    return length


class OWSMProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> OWSMConfig:
        return self.ctx.get_hf_config(OWSMConfig)

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=self.get_hf_config().sample_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_hf_processor(self, **kwargs: object):
        # The multimodal processor consumes waveforms directly, preserving the
        # native ESPnet frontend on the encoder device.
        return self.get_tokenizer()


class OWSMDummyInputsBuilder(BaseDummyInputsBuilder[OWSMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        config = self.info.get_hf_config()
        language = resolve_owsm_language_symbol(
            config.espnet_config["token_list"], "eng"
        )
        return f"<sos>{language}<asr><notimestamps>"

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        config = self.info.get_hf_config()
        return {
            "audio": self._get_dummy_audios(
                length=int(config.max_audio_seconds * config.sample_rate),
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class OWSMMultiModalProcessor(EncDecMultiModalProcessor[OWSMProcessingInfo]):
    skip_decoder_start_token = True

    def create_encoder_prompt(self, prompt, mm_items):
        return [0]

    def _hf_processor_applies_updates(self, *args, **kwargs) -> bool:
        return False

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        config = self.info.get_hf_config()
        audios = mm_data.get("audios", [])
        if isinstance(audios, np.ndarray) and audios.ndim == 1:
            audios = [audios]
        speech = []
        for audio in audios:
            tensor = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
            if tensor.ndim != 1 or not torch.isfinite(tensor).all():
                raise ValueError("OWSM requires finite, mono audio")
            if tensor.numel() > int(config.max_audio_seconds * config.sample_rate):
                raise ValueError(
                    "Audio exceeds max_audio_seconds; segment before inference"
                )
            if config.pad_to_training_window:
                samples = int(config.training_audio_seconds * config.sample_rate)
                if tensor.numel() > samples:
                    raise ValueError("Audio exceeds the checkpoint training window")
                tensor = torch.nn.functional.pad(tensor, (0, samples - tensor.numel()))
            owsm_encoder_length(tensor.numel(), config)
            speech.append(tensor)
        return BatchFeature(
            {
                "input_ids": [[0]],
                "speech": speech,
                "speech_lengths": torch.tensor([x.numel() for x in speech]),
            }
        )

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            "speech": MultiModalFieldConfig.batched("audio"),
            "speech_lengths": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()

        def replacement(item_idx: int):
            item = out_mm_kwargs["audio"][item_idx]
            length = int(item["speech_lengths"].data)
            return [0] * owsm_encoder_length(length, config)

        return [
            PromptReplacement(modality="audio", target=[0], replacement=replacement)
        ]
