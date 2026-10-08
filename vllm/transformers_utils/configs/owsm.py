# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for ESPnet's autoregressive OWSM family."""

from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig


class OWSMConfig(PretrainedConfig):
    model_type = "owsm"

    def __init__(
        self,
        espnet_config=None,
        vocab_size=51865,
        d_model=None,
        decoder_layers=None,
        decoder_attention_heads=None,
        decoder_ffn_dim=None,
        max_position_embeddings=2048,
        max_audio_seconds=30.0,
        pad_to_training_window=True,
        model_version=None,
        **kwargs,
    ):
        native = deepcopy(espnet_config or {})
        encoder = native.get("encoder_conf", {})
        decoder = native.get("decoder_conf", {})
        self.espnet_config = native
        self.vocab_size = vocab_size
        self.d_model = d_model or encoder.get("output_size", 1024)
        self.decoder_layers = decoder_layers or decoder.get("num_blocks", 6)
        self.decoder_attention_heads = decoder_attention_heads or decoder.get(
            "attention_heads", 4
        )
        self.decoder_ffn_dim = decoder_ffn_dim or decoder.get("linear_units", 2048)
        self.hidden_size = self.d_model
        self.num_hidden_layers = self.decoder_layers
        self.num_attention_heads = self.decoder_attention_heads
        self.num_key_value_heads = self.num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.max_audio_seconds = max_audio_seconds
        self.pad_to_training_window = pad_to_training_window
        self.training_audio_seconds = native.get("preprocessor_conf", {}).get(
            "speech_length", max_audio_seconds
        )
        self.model_version = model_version
        self.layer_norm_eps = 1e-12
        self.normalize_before = decoder.get("normalize_before", True)
        self.concat_after = decoder.get("concat_after", False)
        self.qk_norm = decoder.get("qk_norm", False)
        self.sample_rate = native.get("frontend_conf", {}).get("fs", 16000)
        if isinstance(self.sample_rate, str):
            import humanfriendly

            self.sample_rate = humanfriendly.parse_size(self.sample_rate)
        tokens = native.get("token_list", [])
        model = native.get("model_conf", {})
        special_ids = {}
        for key, default in [
            ("bos_token_id", "<sos>"),
            ("eos_token_id", "<eos>"),
            ("pad_token_id", "<blank>"),
        ]:
            symbol = model.get(
                {"bos_token_id": "sym_sos", "eos_token_id": "sym_eos"}.get(
                    key, "sym_blank"
                ),
                default,
            )
            special_ids[key] = kwargs.pop(
                key, tokens.index(symbol) if symbol in tokens else None
            )
        kwargs.pop("is_encoder_decoder", None)
        kwargs.pop("tie_word_embeddings", None)
        decoder_start = kwargs.pop(
            "decoder_start_token_id", special_ids["bos_token_id"]
        )
        super().__init__(
            is_encoder_decoder=True,
            tie_word_embeddings=False,
            decoder_start_token_id=decoder_start,
            **special_ids,
            **kwargs,
        )
