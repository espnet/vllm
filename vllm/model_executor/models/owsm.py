# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Autoregressive OWSM v4/v3.x with native ESPnet encoders and vLLM KV caches."""

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention, CrossAttention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.torch_utils import set_default_torch_dtype

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .owsm_decoder import OWSMDecoder
from .owsm_processing import (
    OWSMDummyInputsBuilder,
    OWSMMultiModalProcessor,
    OWSMProcessingInfo,
)
from .utils import maybe_prefix


class OWSMGlobalMVN(nn.Module):
    """Portable ESPnet GlobalMVN whose statistics come from checkpoint buffers."""

    def __init__(self, input_size: int, config: dict[str, Any]):
        super().__init__()
        self.norm_means = config.get("norm_means", True)
        self.norm_vars = config.get("norm_vars", True)
        self.register_buffer("mean", torch.zeros(input_size, dtype=torch.float32))
        self.register_buffer("std", torch.ones(input_size, dtype=torch.float32))

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        if self.norm_means:
            features = features - self.mean.to(features)
        if self.norm_vars:
            features = features / self.std.to(features)
        mask = torch.arange(features.shape[1], device=features.device)
        mask = mask.unsqueeze(0) >= lengths.unsqueeze(1)
        return features.masked_fill(mask.unsqueeze(-1), 0), lengths


class OWSMEncoder(nn.Module):
    """Unmodified ESPnet encoder layers; decoder/CTC training heads are omitted."""

    def __init__(self, config: Any):
        super().__init__()
        espnet_config = config.espnet_config
        if espnet_config.get("input_size") is not None:
            raise ValueError("OWSM audio inference requires an ESPnet frontend")
        if espnet_config.get("frontend", "default") != "default":
            raise ValueError("OWSM supports the native ESPnet default audio frontend")
        if espnet_config.get("preencoder") is not None:
            raise ValueError("OWSM checkpoints with a preencoder are unsupported")
        if espnet_config.get("postencoder") is not None:
            raise ValueError("OWSM checkpoints with a postencoder are unsupported")
        encoder_conf = dict(espnet_config.get("encoder_conf", {}))
        if encoder_conf.get("interctc_use_conditioning", False):
            raise ValueError("OWSM CTC-conditioned encoders require a separate adapter")
        try:
            from espnet2.asr.frontend.default import DefaultFrontend
            from espnet2.layers.utterance_mvn import UtteranceMVN
        except ImportError as exc:
            raise ImportError(
                "OWSM requires ESPnet. Install an ESPnet version supporting the "
                "checkpoint with `uv pip install espnet`."
            ) from exc

        self.frontend = DefaultFrontend(**espnet_config.get("frontend_conf", {}))
        input_size = self.frontend.output_size()
        normalization = espnet_config.get("normalize", "utterance_mvn")
        normalization_config = espnet_config.get("normalize_conf", {})
        if normalization == "utterance_mvn":
            self.normalize = UtteranceMVN(**normalization_config)
        elif normalization == "global_mvn":
            self.normalize = OWSMGlobalMVN(input_size, normalization_config)
        elif normalization is None:
            self.normalize = None
        else:
            raise ValueError(f"Unsupported OWSM feature normalization: {normalization}")

        encoder_type = espnet_config.get("encoder")
        if encoder_type == "e_branchformer":
            from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder

            encoder_class = EBranchformerEncoder
        elif encoder_type == "transformer":
            from espnet2.asr.encoder.transformer_encoder import TransformerEncoder

            encoder_class = TransformerEncoder
        elif encoder_type == "conformer":
            from espnet2.asr.encoder.conformer_encoder import ConformerEncoder

            encoder_class = ConformerEncoder
        else:
            raise ValueError(
                f"Unsupported autoregressive OWSM encoder: {encoder_type!r}. "
                "OWSM-CTC is not an autoregressive OWSM checkpoint."
            )
        encoder_conf.pop("gradient_checkpoint_layers", None)
        self.encoder = encoder_class(input_size=input_size, **encoder_conf)
        # ESPnet builds sinusoidal caches on CPU in fp32 before model.to().
        # CUDA sin/cos and reduced-precision construction can round differently.
        from espnet2.legacy.nets.pytorch_backend.transformer.embedding import (
            PositionalEncoding,
            RelPositionalEncoding,
        )

        with torch.device("cpu"), set_default_torch_dtype(torch.float32):
            for module in self.encoder.modules():
                if isinstance(module, (PositionalEncoding, RelPositionalEncoding)):
                    length = module.pe.shape[1]
                    if isinstance(module, RelPositionalEncoding):
                        length = (length + 1) // 2
                    module.pe = None
                    module.extend_pe(torch.empty(1, length))
        if self.encoder.output_size() != config.d_model:
            raise ValueError("OWSM encoder and decoder hidden sizes disagree")
        # ESPnet's mel matrix is built with torch.from_numpy, which ignores the
        # device context used by vLLM model initialization.
        encoder_device = next(self.encoder.parameters()).device
        self.frontend.to(device=encoder_device, dtype=torch.float32)

    def forward(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        encoder_parameter = next(self.encoder.parameters())
        # ESPnet computes STFT, log-mel and MVN outside mixed precision.
        with torch.autocast(device_type=speech.device.type, enabled=False):
            features, lengths = self.frontend(speech.float(), speech_lengths)
            if self.normalize is not None:
                features, lengths = self.normalize(features, lengths)
        encoded, output_lengths, _ = self.encoder(
            features.to(encoder_parameter.dtype), lengths
        )
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        return tuple(
            encoded[i, : int(length)] for i, length in enumerate(output_lengths)
        )


@MULTIMODAL_REGISTRY.register_processor(
    OWSMMultiModalProcessor,
    info=OWSMProcessingInfo,
    dummy_inputs=OWSMDummyInputsBuilder,
)
class OWSMForConditionalGeneration(nn.Module, SupportsMultiModal):
    """OWSM AR decoding. Initial implementation supports TP=PP=1, no quantization."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("OWSM currently requires tensor_parallel_size=1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("OWSM currently requires pipeline_parallel_size=1")
        if vllm_config.quant_config is not None:
            raise ValueError("OWSM weight quantization is not implemented")
        if vllm_config.model_config.get_multimodal_config().mm_encoder_only:
            raise ValueError(
                "OWSM currently requires the complete encoder-decoder model"
            )
        self.config = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype
        if self.config.espnet_config.get("model", "espnet") != "espnet":
            raise ValueError("OWSM-CTC requires a separate non-autoregressive engine")
        if self.config.espnet_config.get("model_conf", {}).get("ctc_weight", 0.3) == 1:
            raise ValueError("A CTC-only checkpoint has no autoregressive OWSM decoder")

        def attention_factory(
            num_heads: int, head_size: int, cross_attention: bool, prefix: str
        ):
            attention_type = CrossAttention if cross_attention else Attention
            return attention_type(
                num_heads,
                head_size,
                head_size**-0.5,
                cache_config=vllm_config.cache_config,
                prefix=prefix,
            )

        with self._mark_composite_model(
            vllm_config,
            language_targets=OWSMDecoder,
            tower_targets={"audio": OWSMEncoder},
        ):
            self.encoder = OWSMEncoder(self.config)
            self.decoder = OWSMDecoder(
                self.config,
                attention_factory=attention_factory,
                prefix=maybe_prefix(prefix, "decoder"),
            )
        self.proj_out = ParallelLMHead(
            self.config.vocab_size,
            self.config.d_model,
            bias=True,
            prefix=maybe_prefix(prefix, "proj_out"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.eval()

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None
        raise ValueError("OWSM only supports audio inputs")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder.embed_input_ids(input_ids)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        speech = kwargs.get("speech")
        if speech is None:
            raise ValueError("OWSM requires speech waveforms")
        speech_lengths = kwargs.get("speech_lengths")
        if isinstance(speech, torch.Tensor):
            waveforms = speech.unsqueeze(0) if speech.ndim == 1 else speech
        elif isinstance(speech, (list, tuple)):
            waveforms = speech
        else:
            raise TypeError("OWSM speech must be a tensor or sequence of tensors")
        outputs = []
        device = next(self.encoder.parameters()).device
        # Process each clip without padded neighbours: the encoder mask and memory
        # lengths must describe exactly the placeholder reserved by the scheduler.
        for i, waveform in enumerate(waveforms):
            if not isinstance(waveform, torch.Tensor):
                raise TypeError("OWSM waveform entries must be tensors")
            waveform = waveform.reshape(-1).to(device=device, dtype=torch.float32)
            length = (
                int(speech_lengths[i])
                if isinstance(speech_lengths, (torch.Tensor, Sequence))
                else waveform.numel()
            )
            if not 0 < length <= waveform.numel():
                raise ValueError("OWSM speech_lengths must fit the waveform")
            lengths = torch.tensor([length], dtype=torch.long, device=device)
            outputs.extend(self.encoder(waveform[:length].unsqueeze(0), lengths))
        return tuple(outputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_outputs: list[torch.Tensor] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        encoder_states = torch.cat(encoder_outputs) if encoder_outputs else None
        return self.decoder(input_ids, positions, encoder_states, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(
            self.proj_out, hidden_states, embedding_bias=self.proj_out.bias
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tensors = dict(self.named_parameters())
        tensors.update(self.named_buffers())
        loaded = set()
        fused_shards: dict[str, set[int]] = {}
        for source_name, weight in weights:
            name, shard, shard_count = _map_owsm_weight(source_name)
            if name is None:
                continue
            if name not in tensors and name.startswith("encoder.encoder."):
                # Both checkpoint formats occur in the OWSM releases. Rename only
                # when the installed ESPnet module uses the other layout.
                for old, new in (
                    (".embed.out.0.", ".embed.out."),
                    (".embed.out.1.", ".embed.pos_enc."),
                ):
                    candidate = name.replace(old, new)
                    if candidate in tensors:
                        name = candidate
                        break
                    candidate = name.replace(new, old)
                    if candidate in tensors:
                        name = candidate
                        break
            if name not in tensors:
                raise ValueError(f"Unexpected OWSM checkpoint tensor: {source_name}")
            target = tensors[name]
            if shard is not None:
                fused_shards.setdefault(name, set()).add(shard)
                target = target.chunk(shard_count, dim=0)[shard]
            if target.shape != weight.shape:
                if shard is None and hasattr(target, "weight_loader"):
                    target.weight_loader(target, weight)
                else:
                    raise ValueError(
                        f"OWSM tensor {source_name} shape {tuple(weight.shape)} "
                        f"does not match {tuple(target.shape)}"
                    )
            else:
                with torch.no_grad():
                    target.copy_(weight)
            if shard is None:
                loaded.add(name)
        for name, shards in fused_shards.items():
            expected = 3 if ".qkv_proj." in name else 2
            if len(shards) != expected:
                raise ValueError(
                    f"OWSM checkpoint is missing projection shards: {name}"
                )
            loaded.add(name)
        # Statistics are buffers, but silently keeping their defaults changes logits.
        if isinstance(self.encoder.normalize, OWSMGlobalMVN):
            required = {"encoder.normalize.mean", "encoder.normalize.std"}
            if not required.issubset(loaded):
                raise ValueError(
                    "OWSM GlobalMVN statistics are missing from checkpoint"
                )
        missing = set(dict(self.named_parameters())) - loaded
        if missing:
            raise ValueError(f"OWSM checkpoint is missing model parameters: {missing}")
        if "encoder.frontend.logmel.melmat" not in loaded:
            raise ValueError(
                "OWSM checkpoint is missing its frontend mel-filter buffer"
            )
        return loaded


def _map_owsm_weight(name: str) -> tuple[str | None, int | None, int]:
    """Map native ESPnet names, retaining every bias and projection shard."""
    if name.startswith("module."):
        name = name.removeprefix("module.")
    if name.startswith(("ctc.", "criterion_att.", "specaug.")):
        return None, None, 1
    if name.endswith(".pe"):
        return None, None, 1
    if name.startswith("decoder.output_layer."):
        return name.replace("decoder.output_layer.", "proj_out.", 1), None, 1
    if name.startswith(("frontend.", "normalize.", "encoder.")):
        name = f"encoder.{name}"
    for projection, shard in (("q", 0), ("k", 1), ("v", 2)):
        source = f".self_attn.linear_{projection}."
        if name.startswith("decoder.") and source in name:
            return name.replace(source, ".self_attn.qkv_proj."), shard, 3
    if name.startswith("decoder."):
        if ".src_attn.linear_q." in name:
            return name.replace(".src_attn.linear_q.", ".src_attn.q_proj."), None, 1
        for projection, shard in (("k", 0), ("v", 1)):
            source = f".src_attn.linear_{projection}."
            if source in name:
                return name.replace(source, ".src_attn.kv_proj."), shard, 2
    return name, None, 1
