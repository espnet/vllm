# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare OWSM decoder math and cached decoding against the native ESPnet model.

The CPU attention below tests projection, positional and layer math. GPU engine
tests must separately exercise the production vLLM paged-attention backend.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.owsm_decoder import OWSMDecoder


class CachedReferenceAttention(nn.Module):
    def __init__(self, heads: int, head_dim: int, cross: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.cross = cross
        self.key = self.value = None

    def forward(self, query, key, value):
        prefix = 0
        if key is not None:
            if self.cross or self.key is None:
                self.key, self.value = key, value
            else:
                prefix = self.key.shape[0]
                self.key = torch.cat((self.key, key))
                self.value = torch.cat((self.value, value))
        assert self.key is not None and self.value is not None
        key, value = self.key, self.value
        query = query.reshape(-1, self.heads, self.head_dim).transpose(0, 1)
        key = key.reshape(-1, self.heads, self.head_dim).transpose(0, 1)
        value = value.reshape(-1, self.heads, self.head_dim).transpose(0, 1)
        scores = query @ key.transpose(-1, -2) / self.head_dim**0.5
        if not self.cross:
            q_positions = torch.arange(query.shape[1]) + prefix
            k_positions = torch.arange(key.shape[1])
            allowed = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        output = torch.softmax(scores, dim=-1) @ value
        return output.transpose(0, 1).reshape(-1, self.heads * self.head_dim)


def reference_factory(heads, head_dim, cross, prefix):
    return CachedReferenceAttention(heads, head_dim, cross)


def copy_native_decoder_weights(decoder, native):
    own = dict(decoder.named_parameters())
    for name, tensor in native.state_dict().items():
        if name.startswith("output_layer."):
            continue
        shard = None
        shard_count = 1
        for projection, index in (("q", 0), ("k", 1), ("v", 2)):
            source = f".self_attn.linear_{projection}."
            if source in name:
                name = name.replace(source, ".self_attn.qkv_proj.")
                shard, shard_count = index, 3
                break
        name = name.replace(".src_attn.linear_q.", ".src_attn.q_proj.")
        for projection, index in (("k", 0), ("v", 1)):
            source = f".src_attn.linear_{projection}."
            if source in name:
                name = name.replace(source, ".src_attn.kv_proj.")
                shard, shard_count = index, 2
                break
        target = own[name]
        if shard is not None:
            target = target.chunk(shard_count, dim=0)[shard]
        target.data.copy_(tensor)


@pytest.mark.parametrize(
    "normalize_before,concat_after,qk_norm",
    [
        (True, False, False),
        (True, False, True),
        (False, False, False),
        (True, True, False),
    ],
)
@torch.inference_mode()
def test_logits_and_incremental_cache_match_espnet(
    normalize_before, concat_after, qk_norm
):
    """Guard bias, eps=1e-12, scaled positions, QKV layout and cross-cache reuse."""
    espnet = pytest.importorskip("espnet2.asr.decoder.transformer_decoder")
    torch.manual_seed(9)
    native = espnet.TransformerDecoder(
        vocab_size=37,
        encoder_output_size=32,
        attention_heads=4,
        linear_units=64,
        num_blocks=2,
        dropout_rate=0.0,
        positional_dropout_rate=0.0,
        use_flash_attn=False,
        qk_norm=qk_norm,
        normalize_before=normalize_before,
        concat_after=concat_after,
    ).eval()
    config = SimpleNamespace(
        espnet_config={
            "decoder": "transformer",
            "decoder_conf": {
                "normalize_before": normalize_before,
                "concat_after": concat_after,
                "qk_norm": qk_norm,
            },
        },
        vocab_size=37,
        d_model=32,
        decoder_attention_heads=4,
        decoder_ffn_dim=64,
        decoder_layers=2,
        max_position_embeddings=64,
    )
    decoder = OWSMDecoder(config, attention_factory=reference_factory).eval()
    copy_native_decoder_weights(decoder, native)
    tokens = torch.tensor([2, 7, 12, 19, 3, 5])
    encoder = torch.randn(1, 9, 32)
    (expected, hidden), _ = native(
        encoder,
        torch.tensor([9]),
        tokens.unsqueeze(0),
        torch.tensor([6]),
        return_hs=True,
    )
    actual_hidden = decoder(tokens, torch.arange(6), encoder[0])
    actual = native.output_layer(actual_hidden)
    torch.testing.assert_close(actual_hidden, hidden[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual, expected[0], rtol=1e-5, atol=1e-6)

    incremental = OWSMDecoder(config, attention_factory=reference_factory).eval()
    copy_native_decoder_weights(incremental, native)
    prefix = incremental(tokens[:3], torch.arange(3), encoder[0])
    steps = [prefix]
    for position in range(3, 6):
        steps.append(
            incremental(tokens[position : position + 1], torch.tensor([position]), None)
        )
    torch.testing.assert_close(torch.cat(steps), actual_hidden, rtol=1e-5, atol=1e-6)
