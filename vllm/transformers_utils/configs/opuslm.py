# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for OpusLM (OLMo-2-7B based multimodal
speech-language model with 9-stream delay-interleaved discrete codec output).

Vocab layout:
    [0,   256)    - Special tokens (pad=0, bos=1, eos=5, ...)
    [256, 5256)   - SSL tokens (XEUS + K-means, 5000 clusters)
    [5256, 13448) - DAC codec tokens (8 streams × 1024 tokens each)
    [13448, ~)    - Text BPE tokens
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OpusLMConfig(PretrainedConfig):
    """Configuration for OpusLMForConditionalGeneration.

    OpusLM is an OLMo-2-7B based multimodal speech-language model that uses:
    - 1 SSL stream (XEUS + K-means, 5000 clusters) for stream 0
    - 8 DAC codec streams (1024 codes each) for streams 1-8
    - 9-stream delay interleaving during generation
    - No continuous audio encoder (audio input is pre-tokenized)
    """

    model_type = "opuslm"

    def __init__(
        self,
        # OLMo-2 backbone arch
        vocab_size: int = 113870,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        tie_word_embeddings: bool = False,
        # Special token IDs
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 5,
        sos_eos_token_id: int = 5,
        # Token range boundaries
        ssl_token_start: int = 256,
        ssl_token_end: int = 5256,
        codec_token_start: int = 5256,
        codec_token_end: int = 13448,
        text_token_start: int = 13448,
        text_token_end: int = 113800,
        # Modality boundary tokens
        codec_ssl_start_end_token_id: int = 34,   # <codec_ssl_start/end>
        text_bpe_start_end_token_id: int = 35,    # <text_bpe_start/end>
        spk_start_end_token_id: int = 37,         # <spk_start/end>
        # Task identifiers
        textlm_task_token_id: int = 64,           # <textlm_task>
        codec_ssl_asr_task_token_id: int = 80,    # <codec_ssl_asr_task>
        codec_ssl_tts_task_token_id: int = 81,    # <codec_ssl_tts_task>
        codec_ssl_plain_tts_task_token_id: int = 82,   # <codec_ssl_plain_tts_task>
        codec_ssl_audiolm_task_token_id: int = 83,     # <codec_ssl_audiolm_task>
        # Dialogue tokens (same IDs as dialogue model)
        system_prompt_token_id: int = 8,
        user_input_token_id: int = 9,
        assistant_output_token_id: int = 10,
        eou_token_id: int = 11,
        audio_dialogue_task_token_id: int = 89,
        text_dialogue_task_token_id: int = 88,
        speaker_prompt_length: int = 500,
        # Multi-stream settings
        nq: int = 9,              # total streams (1 SSL + 8 DAC)
        num_codec_streams: int = 8,
        codec_per_stream_size: int = 1024,
        # Audio generation parameters
        audio_temperature: float = 0.7,
        audio_topk: int = 30,
        audio_minlen: int = 50,
        # Codec/SSL model tags
        dac_hf_model_tag: str = "ftshijt/espnet_codec_dac_large_v1.4_360epoch",
        xeus_hf_model_tag: str = "espnet/xeus",
        xeus_checkpoint_filename: str = "model/xeus_checkpoint_new.pth",
        km_model_filename: str = "model/km_opus_lm.mdl",
        xeus_layer: int = 18,
        dac_sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # OLMo-2 backbone
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta

        # Token ranges
        self.ssl_token_start = ssl_token_start
        self.ssl_token_end = ssl_token_end
        self.codec_token_start = codec_token_start
        self.codec_token_end = codec_token_end
        self.text_token_start = text_token_start
        self.text_token_end = text_token_end

        # Modality markers
        self.codec_ssl_start_end_token_id = codec_ssl_start_end_token_id
        self.text_bpe_start_end_token_id = text_bpe_start_end_token_id
        self.spk_start_end_token_id = spk_start_end_token_id
        self.sos_eos_token_id = sos_eos_token_id
        self.textlm_task_token_id = textlm_task_token_id
        self.codec_ssl_asr_task_token_id = codec_ssl_asr_task_token_id
        self.codec_ssl_tts_task_token_id = codec_ssl_tts_task_token_id
        self.codec_ssl_plain_tts_task_token_id = codec_ssl_plain_tts_task_token_id
        self.codec_ssl_audiolm_task_token_id = codec_ssl_audiolm_task_token_id

        # Dialogue tokens
        self.system_prompt_token_id = system_prompt_token_id
        self.user_input_token_id = user_input_token_id
        self.assistant_output_token_id = assistant_output_token_id
        self.eou_token_id = eou_token_id
        self.audio_dialogue_task_token_id = audio_dialogue_task_token_id
        self.text_dialogue_task_token_id = text_dialogue_task_token_id
        self.speaker_prompt_length = speaker_prompt_length

        # Multi-stream
        self.nq = nq
        self.num_codec_streams = num_codec_streams
        self.codec_per_stream_size = codec_per_stream_size

        # Audio generation
        self.audio_temperature = audio_temperature
        self.audio_topk = audio_topk
        self.audio_minlen = audio_minlen

        # External model tags
        self.dac_hf_model_tag = dac_hf_model_tag
        self.xeus_hf_model_tag = xeus_hf_model_tag
        self.xeus_checkpoint_filename = xeus_checkpoint_filename
        self.km_model_filename = km_model_filename
        self.xeus_layer = xeus_layer
        self.dac_sample_rate = dac_sample_rate


@dataclass(frozen=True)
class OpusLMTaskLayout:
    """The ESPnet OpusLM prompt layout, over plain integer token IDs.

    An OpusLM prompt is not a bare tokenization of the user text. ESPnet wraps
    it as ``<sos/eos> <task_token> <condition segment> <target prefix>``, and
    the model was only ever trained on that shape.

    Two different places have to build it. Requests that carry audio go through
    ``OpusLMMultiModalProcessor``, which has the model config in hand. Requests
    that do not carry audio never reach that processor at all -- vLLM's input
    preprocessor returns the tokenized prompt untouched when there is no
    multimodal data -- so for those the only remaining hook is the tokenizer
    wrapper, which has no model config. Keeping the layout here, over ints,
    lets both sides produce the same sequence without either importing the
    other (and without pulling torch into the tokenizer).

    ``apply`` and ``strip`` are exact inverses, so it is safe for the tokenizer
    to lay a prompt out and for the multimodal processor to later strip that
    layout and redo it with the task token it resolved itself.
    """

    sos_eos_token_id: int = 5
    pad_token_id: int = 0
    codec_ssl_start_end_token_id: int = 34
    text_bpe_start_end_token_id: int = 35
    textlm_task_token_id: int = 64
    codec_ssl_asr_task_token_id: int = 80
    codec_ssl_tts_task_token_id: int = 81
    codec_ssl_plain_tts_task_token_id: int = 82
    codec_ssl_audiolm_task_token_id: int = 83
    text_dialogue_task_token_id: int = 88
    audio_dialogue_task_token_id: int = 89
    system_prompt_token_id: int = 8
    user_input_token_id: int = 9
    assistant_output_token_id: int = 10
    nq: int = 9

    @classmethod
    def from_config(cls, config) -> "OpusLMTaskLayout":
        """Read the layout out of an OpusLM (or OpusLM-dialogue) config.

        Every field falls back to this dataclass' own default, which matches the
        released checkpoints, so a config that predates one of these keys still
        produces a usable layout.
        """
        kwargs = {}
        for field in fields(cls):
            default = field.default
            if field.name == "sos_eos_token_id":
                # ESPnet shares one token between <sos> and <eos>.
                default = int(getattr(config, "eos_token_id", default) or default)
            value = getattr(config, field.name, None)
            kwargs[field.name] = int(default if value is None else value)
        return cls(**kwargs)

    @property
    def known_task_token_ids(self) -> frozenset[int]:
        """Every token ID that can legally sit at position 1 of a prompt."""
        return frozenset(
            {
                self.textlm_task_token_id,
                self.codec_ssl_asr_task_token_id,
                self.codec_ssl_tts_task_token_id,
                self.codec_ssl_plain_tts_task_token_id,
                self.codec_ssl_audiolm_task_token_id,
                self.text_dialogue_task_token_id,
                self.audio_dialogue_task_token_id,
            }
        )

    @property
    def audio_out_task_token_ids(self) -> frozenset[int]:
        """Tasks whose *target* segment is codec frames, i.e. the TTS tasks."""
        return frozenset(
            {
                self.codec_ssl_tts_task_token_id,
                self.codec_ssl_plain_tts_task_token_id,
            }
        )

    def resolve_task_token_id(
        self,
        *,
        has_audio_input: bool,
        mode: str | None = None,
        task: str | int | None = None,
    ) -> int:
        """Pick the task token for a request.

        An explicit ``task`` wins; otherwise the request ``mode`` decides; with
        neither, the presence of audio input decides (audio in means ASR, no
        audio means plain TTS).
        """
        if isinstance(task, bool):
            raise TypeError("OpusLM task must be a string or a token ID, not a bool")
        if isinstance(task, int):
            return int(task)

        task_aliases = {
            "asr": self.codec_ssl_asr_task_token_id,
            "codec_ssl_asr": self.codec_ssl_asr_task_token_id,
            "codec_ssl_asr_task": self.codec_ssl_asr_task_token_id,
            "tts": self.codec_ssl_tts_task_token_id,
            "codec_ssl_tts": self.codec_ssl_tts_task_token_id,
            "codec_ssl_tts_task": self.codec_ssl_tts_task_token_id,
            "plain_tts": self.codec_ssl_plain_tts_task_token_id,
            "codec_ssl_plain_tts": self.codec_ssl_plain_tts_task_token_id,
            "codec_ssl_plain_tts_task": self.codec_ssl_plain_tts_task_token_id,
            "textlm": self.textlm_task_token_id,
            "text_lm": self.textlm_task_token_id,
            "olmo_textlm": self.textlm_task_token_id,
            "textlm_task": self.textlm_task_token_id,
            "audio_dialogue": self.audio_dialogue_task_token_id,
            "audio_dialogue_task": self.audio_dialogue_task_token_id,
            "text_dialogue": self.text_dialogue_task_token_id,
            "text_dialogue_task": self.text_dialogue_task_token_id,
        }
        if isinstance(task, str):
            task_norm = task.strip().lower()
            if task_norm in task_aliases:
                return task_aliases[task_norm]
            raise ValueError(
                f"Unsupported OpusLM task '{task}'. "
                "Supported: asr, tts, plain_tts, textlm, "
                "audio_dialogue, text_dialogue."
            )

        mode_norm = mode.strip().lower() if isinstance(mode, str) else None
        if mode_norm is not None and mode_norm not in MODE_TO_TASK_ALIAS:
            raise ValueError(
                f"Unsupported OpusLM mode '{mode}'. "
                f"Supported: {', '.join(sorted(MODE_TO_TASK_ALIAS))}."
            )

        if mode_norm == "audio_dialogue":
            return self.audio_dialogue_task_token_id
        if mode_norm == "text_dialogue":
            return self.text_dialogue_task_token_id
        if mode_norm == "audio_text":
            return self.codec_ssl_asr_task_token_id
        if mode_norm == "text_text":
            return self.textlm_task_token_id
        if mode_norm == "text_audio":
            # A TTS request that also carries audio is voice cloning, which is
            # the non-plain TTS task.
            if has_audio_input:
                return self.codec_ssl_tts_task_token_id
            return self.codec_ssl_plain_tts_task_token_id
        if has_audio_input:
            return self.codec_ssl_asr_task_token_id
        return self.codec_ssl_plain_tts_task_token_id

    def apply(self, body_ids: list[int], task_token_id: int) -> list[int]:
        """Wrap an already-laid-out-or-bare body in the ESPnet task layout.

        ``body_ids`` must already be in the model's global ID space (the
        OpusLM tokenizer shifts text BPE IDs by ``text_token_start`` when it
        encodes, so no further offset is applied here). If ``body_ids`` still
        carries a previous layout, it is stripped first, which makes this
        method idempotent.
        """
        task_token_id = int(task_token_id)
        body = self.strip(body_ids)
        seq = [self.sos_eos_token_id, task_token_id, *body]

        if task_token_id in self.audio_out_task_token_ids:
            # Condition segment: <text_bpe_start/end> then the text to speak.
            # Added unconditionally, and `strip` removes exactly one, so the two
            # stay exact inverses even for a body that itself begins with that
            # marker (a bare text body never legitimately does: the marker is
            # ESPnet's text delimiter, not part of the text).
            seq = [
                self.sos_eos_token_id,
                task_token_id,
                self.text_bpe_start_end_token_id,
                *body,
            ]
            # ESPnet pads nq-1 zero frames between the text condition and the
            # codec target so delay interleaving cannot overlap the two.
            seq.extend([self.pad_token_id] * (self.nq - 1))
            # Target segment prefix.
            seq.append(self.codec_ssl_start_end_token_id)
        elif task_token_id == self.codec_ssl_asr_task_token_id:
            # Target text segment prefix.
            seq.append(self.text_bpe_start_end_token_id)

        return seq

    def apply_dialogue(
        self,
        turns: Sequence[tuple[str, Sequence[int]]],
        task_token_id: int,
    ) -> list[int]:
        """Lay out a dialogue prompt whose turns are all text.

        ``turns`` is the conversation as ``(role, body_ids)`` pairs, already in
        the model's global ID space. The generation target is appended here
        rather than taken from ``turns``: ESPnet marks it with the two tokens
        ``<role> <modality>`` and nothing else, and the request never carries a
        turn for text it has not produced yet.

        The target's modality comes from the task and the turn that precedes it.
        A text dialogue always targets text. An audio dialogue turn is two
        segments in ESPnet -- the assistant writes the reply as text first, then
        speaks it as codec frames -- so the target is codec only when the turn
        immediately before it is the assistant's own text, and text otherwise.

        Dialogue turns that carry audio are laid out by the model's multimodal
        processor instead, which owns speaker prompts, audio placeholders and
        multipart content. This method exists because a text-only request never
        reaches that processor -- vLLM's input preprocessor forwards a prompt
        with no multimodal data untouched -- so without it a text dialogue would
        arrive as bare BPE with no task token and no role markers.
        """
        role_tokens = {
            "system": self.system_prompt_token_id,
            "user": self.user_input_token_id,
            "assistant": self.assistant_output_token_id,
        }
        inter_pad = self.nq - 1
        seq = [self.sos_eos_token_id, int(task_token_id)]
        for role, body_ids in turns:
            seq.append(role_tokens.get(role, self.user_input_token_id))
            seq.append(self.text_bpe_start_end_token_id)
            seq.extend(int(tid) for tid in body_ids)
            # Non-target segments carry no end token, and ESPnet pads nq-1 zero
            # frames after each one so delay interleaving cannot overlap two
            # segments.
            seq.extend([self.pad_token_id] * inter_pad)

        speaks_next = (
            int(task_token_id) != self.text_dialogue_task_token_id
            and len(turns) > 0
            and turns[-1][0] == "assistant"
            and len(turns[-1][1]) > 0
        )
        seq.append(self.assistant_output_token_id)
        seq.append(
            self.codec_ssl_start_end_token_id
            if speaks_next
            else self.text_bpe_start_end_token_id
        )
        return seq

    def strip(self, seq: list[int] | tuple[int, ...]) -> list[int]:
        """Inverse of ``apply``: recover the bare body from a laid-out prompt.

        A sequence that was never laid out is returned unchanged, so callers
        can strip unconditionally.
        """
        ids = [int(t) for t in seq]
        if len(ids) < 2:
            return ids
        if ids[0] != self.sos_eos_token_id or ids[1] not in self.known_task_token_ids:
            return ids

        task_token_id = ids[1]
        body = ids[2:]
        if task_token_id in self.audio_out_task_token_ids:
            if body and body[-1] == self.codec_ssl_start_end_token_id:
                body = body[:-1]
            inter_pad = self.nq - 1
            if len(body) >= inter_pad and all(
                t == self.pad_token_id for t in body[-inter_pad:]
            ):
                body = body[:-inter_pad]
            if body and body[0] == self.text_bpe_start_end_token_id:
                body = body[1:]
        elif task_token_id == self.codec_ssl_asr_task_token_id:
            if body and body[-1] == self.text_bpe_start_end_token_id:
                body = body[:-1]
        return body


# Request ``mode`` values, and the task alias each one means. Kept next to the
# layout because the client, the tokenizer and the model runner all key off the
# same strings.
MODE_TO_TASK_ALIAS = {
    "text_audio": "plain_tts",
    "audio_text": "asr",
    "text_text": "textlm",
    "audio_dialogue": "audio_dialogue",
    "text_dialogue": "text_dialogue",
}

# Request modes whose response contains a codec stream, and therefore ends with
# the ARDelay flush tail that the scheduler's EOS deferral is about.
AUDIO_OUT_MODES = frozenset({"text_audio", "audio_dialogue"})


__all__ = [
    "AUDIO_OUT_MODES",
    "MODE_TO_TASK_ALIAS",
    "OpusLMConfig",
    "OpusLMTaskLayout",
]
