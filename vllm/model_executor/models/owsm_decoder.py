# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ESPnet Transformer decoder math, with an injected attention implementation."""

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

AttentionFactory = Callable[[int, int, bool, str], nn.Module]


class OWSMPositionalEncoding(nn.Module):
    """ESPnet's interleaved sinusoidal positions and embedding scaling."""

    def __init__(self, d_model: int, max_positions: int, scaled: bool = False):
        super().__init__()
        if d_model % 2:
            raise ValueError("OWSM requires an even decoder hidden size")
        positions = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        table = torch.empty(max_positions, d_model, dtype=torch.float32)
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("pe", table, persistent=False)
        self.xscale = 1.0 if scaled else math.sqrt(d_model)
        if scaled:
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("alpha", None)

    def forward(self, embeddings: torch.Tensor, positions: torch.Tensor):
        positional = self.pe[positions].to(embeddings.dtype)
        if self.alpha is not None:
            positional = positional * self.alpha
        return embeddings * self.xscale + positional


class OWSMAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qk_norm: bool,
        cross_attention: bool,
        attention_factory: AttentionFactory,
        prefix: str,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("OWSM hidden size must be divisible by attention heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.cross_attention = cross_attention
        if cross_attention:
            self.q_proj = nn.Linear(d_model, d_model)
            self.kv_proj = nn.Linear(d_model, 2 * d_model)
        else:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.q_norm = (
            nn.LayerNorm(self.head_dim, eps=1e-12) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.LayerNorm(self.head_dim, eps=1e-12) if qk_norm else nn.Identity()
        )
        self.attn = attention_factory(
            num_heads, self.head_dim, cross_attention, f"{prefix}.attn"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cross_attention:
            query = self.q_proj(hidden_states)
            if encoder_hidden_states is None:
                key = value = None
            else:
                key, value = self.kv_proj(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)

        query = self.q_norm(query.reshape(-1, self.num_heads, self.head_dim))
        query = query.reshape(-1, self.d_model)
        if key is not None:
            key = self.k_norm(key.reshape(-1, self.num_heads, self.head_dim))
            key = key.reshape(-1, self.d_model)
        attended = self.attn(query, key, value)
        return self.linear_out(attended)


class OWSMFeedForward(nn.Module):
    def __init__(self, d_model: int, linear_units: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, linear_units)
        self.w_2 = nn.Linear(linear_units, d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w_2(torch.relu(self.w_1(hidden_states)))


class OWSMDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        attention_factory: AttentionFactory,
        prefix: str,
    ):
        super().__init__()
        decoder_conf = config.espnet_config.get("decoder_conf", {})
        self.normalize_before = decoder_conf.get("normalize_before", True)
        self.concat_after = decoder_conf.get("concat_after", False)
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-12)
        self.norm3 = nn.LayerNorm(config.d_model, eps=1e-12)
        qk_norm = decoder_conf.get("qk_norm", False)
        self.self_attn = OWSMAttention(
            config.d_model,
            config.decoder_attention_heads,
            qk_norm,
            False,
            attention_factory,
            f"{prefix}.self_attn",
        )
        self.src_attn = OWSMAttention(
            config.d_model,
            config.decoder_attention_heads,
            qk_norm,
            True,
            attention_factory,
            f"{prefix}.src_attn",
        )
        self.feed_forward = OWSMFeedForward(config.d_model, config.decoder_ffn_dim)
        if self.concat_after:
            self.concat_linear1 = nn.Linear(2 * config.d_model, config.d_model)
            self.concat_linear2 = nn.Linear(2 * config.d_model, config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = (
            self.norm1(hidden_states) if self.normalize_before else hidden_states
        )
        attended = self.self_attn(normalized)
        if self.concat_after:
            attended = self.concat_linear1(torch.cat((normalized, attended), dim=-1))
        hidden_states = residual + attended
        if not self.normalize_before:
            hidden_states = self.norm1(hidden_states)

        residual = hidden_states
        normalized = (
            self.norm2(hidden_states) if self.normalize_before else hidden_states
        )
        attended = self.src_attn(normalized, encoder_hidden_states)
        if self.concat_after:
            attended = self.concat_linear2(torch.cat((normalized, attended), dim=-1))
        hidden_states = residual + attended
        if not self.normalize_before:
            hidden_states = self.norm2(hidden_states)

        residual = hidden_states
        normalized = (
            self.norm3(hidden_states) if self.normalize_before else hidden_states
        )
        hidden_states = residual + self.feed_forward(normalized)
        if not self.normalize_before:
            hidden_states = self.norm3(hidden_states)
        return hidden_states


class OWSMDecoder(nn.Module):
    """Packed token decoder; production attention and KV caches belong to vLLM."""

    def __init__(
        self,
        config: Any,
        *,
        attention_factory: AttentionFactory,
        prefix: str = "decoder",
    ):
        super().__init__()
        decoder_conf = config.espnet_config.get("decoder_conf", {})
        if config.espnet_config.get("decoder", "transformer") != "transformer":
            raise ValueError(
                "OWSM vLLM requires the autoregressive Transformer decoder"
            )
        if decoder_conf.get("input_layer", "embed") != "embed":
            raise ValueError("OWSM vLLM requires decoder input_layer='embed'")
        if not decoder_conf.get("use_output_layer", True):
            raise ValueError("OWSM vLLM requires the decoder output projection")
        positional_type = decoder_conf.get("pos_enc_class", "PositionalEncoding")
        if positional_type not in ("PositionalEncoding", "ScaledPositionalEncoding"):
            raise ValueError(f"Unsupported OWSM decoder positions: {positional_type}")
        self.embed = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size, config.d_model),
                OWSMPositionalEncoding(
                    config.d_model,
                    config.max_position_embeddings,
                    scaled=positional_type == "ScaledPositionalEncoding",
                ),
            ]
        )
        self.decoders = nn.ModuleList(
            OWSMDecoderLayer(config, attention_factory, f"{prefix}.decoders.{i}")
            for i in range(config.decoder_layers)
        )
        self.normalize_before = decoder_conf.get("normalize_before", True)
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(config.d_model, eps=1e-12)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed[0](input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.embed[1](inputs_embeds, positions)
        for layer in self.decoders:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        if self.normalize_before:
            hidden_states = self.after_norm(hidden_states)
        return hidden_states
