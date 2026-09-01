# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tokenizer wrapper for OpusLM global token layout.

OpusLM reserves low token IDs for non-text modalities and special markers:
  [0, 13448) are not plain text BPE IDs.

Text BPE IDs from the base tokenizer are shifted by +text_token_offset when
encoding, and shifted back when decoding.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

from transformers import BatchEncoding

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.transformers_utils.configs.opuslm import OpusLMTaskLayout

from .hf import CachedHfTokenizer
from .protocol import TokenizerLike

# What an audio item looks like in a conversation by the time the renderer calls
# apply_chat_template: with the "openai" content format the parts survive as
# dicts, and with the "string" format each one has already been replaced by the
# model's placeholder string (OpusLMForCausalLM.get_placeholder_str).
_AUDIO_PART_TYPES = frozenset({"input_audio", "audio_url", "audio"})
_AUDIO_PLACEHOLDER_STRS = ("<codec_ssl_start_end>", "<codec_ssl_start/end>")


def _conversation_has_audio(messages: Sequence[Any]) -> bool:
    """Whether any message in the conversation carries an audio item."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            if any(ph in content for ph in _AUDIO_PLACEHOLDER_STRS):
                return True
        elif isinstance(content, Sequence):
            for part in content:
                if isinstance(part, dict) and part.get("type") in _AUDIO_PART_TYPES:
                    return True
    return False


class OpusLMTokenizer(CachedHfTokenizer):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "TokenizerLike":
        text_token_offset = int(kwargs.pop("opuslm_text_token_offset", 13448))
        text_token_end = int(kwargs.pop("opuslm_text_token_end", 113800))
        pad_token_id = int(kwargs.pop("opuslm_pad_token_id", 0))
        eos_token_id = int(kwargs.pop("opuslm_eos_token_id", 5))
        codec_ssl_start_end_token_id = int(
            kwargs.pop("opuslm_codec_ssl_start_end_token_id", 34)
        )
        text_bpe_start_end_token_id = int(
            kwargs.pop("opuslm_text_bpe_start_end_token_id", 35)
        )
        # The task layout the tokenizer lays over text-only prompts. Every id
        # here is also a field of OpusLMTaskLayout, whose own defaults match the
        # shipped checkpoint, so a missing kwarg degrades to the same value.
        task_layout = OpusLMTaskLayout(
            sos_eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            codec_ssl_start_end_token_id=codec_ssl_start_end_token_id,
            text_bpe_start_end_token_id=text_bpe_start_end_token_id,
            **{
                field: int(kwargs.pop(f"opuslm_{field}", default))
                for field, default in (
                    ("textlm_task_token_id", 64),
                    ("codec_ssl_asr_task_token_id", 80),
                    ("codec_ssl_tts_task_token_id", 81),
                    ("codec_ssl_plain_tts_task_token_id", 82),
                    ("codec_ssl_audiolm_task_token_id", 83),
                    ("text_dialogue_task_token_id", 88),
                    ("audio_dialogue_task_token_id", 89),
                    ("nq", 9),
                )
            },
        )

        tokenizer = super().from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )
        return OpusLMTokenizer(
            tokenizer=tokenizer,
            text_token_offset=text_token_offset,
            text_token_end=text_token_end,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            codec_ssl_start_end_token_id=codec_ssl_start_end_token_id,
            text_bpe_start_end_token_id=text_bpe_start_end_token_id,
            task_layout=task_layout,
        )

    def __init__(
        self,
        tokenizer: TokenizerLike,
        *,
        text_token_offset: int,
        text_token_end: int,
        pad_token_id: int,
        eos_token_id: int,
        codec_ssl_start_end_token_id: int,
        text_bpe_start_end_token_id: int,
        task_layout: OpusLMTaskLayout | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.name_or_path = getattr(tokenizer, "name_or_path", "")

        self.text_token_offset = int(text_token_offset)
        self.text_token_end = int(text_token_end)

        self._pad_token_id = int(pad_token_id)
        self._eos_token_id = int(eos_token_id)
        self._bos_token_id = int(eos_token_id)
        self._codec_ssl_start_end_token_id = int(codec_ssl_start_end_token_id)
        self._text_bpe_start_end_token_id = int(text_bpe_start_end_token_id)

        # Callers that build the tokenizer directly (tests) may omit the layout;
        # fall back to one carrying the ids we do have plus the shipped defaults.
        self._task_layout = task_layout or OpusLMTaskLayout(
            sos_eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
            codec_ssl_start_end_token_id=self._codec_ssl_start_end_token_id,
            text_bpe_start_end_token_id=self._text_bpe_start_end_token_id,
        )

        self._special_id_to_token: dict[int, str] = {
            self._pad_token_id: "<pad>",
            self._eos_token_id: "<sos/eos>",
            8: "<system_prompt>",
            9: "<user_input>",
            10: "<assistant_output>",
            11: "<eou>",
            self._codec_ssl_start_end_token_id: "<codec_ssl_start/end>",
            self._text_bpe_start_end_token_id: "<text_bpe_start/end>",
            37: "<spk_start/end>",
            88: "<text_dialogue_task>",
            89: "<audio_dialogue_task>",
        }
        self._special_token_to_id = {v: k for k, v in self._special_id_to_token.items()}

        base_vocab = self.tokenizer.get_vocab()
        self._vocab = {tok: idx + self.text_token_offset for tok, idx in base_vocab.items()}
        for sid, stok in self._special_id_to_token.items():
            self._vocab.setdefault(stok, sid)

        base_added_vocab = self.tokenizer.get_added_vocab()
        self._added_vocab = {
            tok: idx + self.text_token_offset for tok, idx in base_added_vocab.items()
        }

        shifted_base_special_ids = {
            sid + self.text_token_offset for sid in self.tokenizer.all_special_ids
        }
        self._all_special_ids = sorted(
            set(self._special_id_to_token.keys()) | shifted_base_special_ids
        )
        self._all_special_tokens = list(
            dict.fromkeys(
                [*self._special_id_to_token.values(), *self.tokenizer.all_special_tokens]
            )
        )

    def _shift_ids_up(self, input_ids: Any) -> Any:
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return [[int(t) + self.text_token_offset for t in row]
                        for row in input_ids]
            return [int(t) + self.text_token_offset for t in input_ids]
        try:
            return input_ids + self.text_token_offset
        except TypeError:
            return input_ids

    def _is_text_global_id(self, token_id: int) -> bool:
        return self.text_token_offset <= token_id < self.text_token_end

    def _to_base_id(self, token_id: int) -> int:
        return int(token_id) - self.text_token_offset

    def _decode_text_chunk(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        if not token_ids:
            return ""
        base_ids = [self._to_base_id(tid) for tid in token_ids]
        return self.tokenizer.decode(base_ids, skip_special_tokens=skip_special_tokens)

    def num_special_tokens_to_add(self) -> int:
        return self.tokenizer.num_special_tokens_to_add()

    @property
    def all_special_tokens(self) -> list[str]:
        return self._all_special_tokens

    @property
    def all_special_ids(self) -> list[int]:
        return self._all_special_ids

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def is_fast(self) -> bool:
        return self.tokenizer.is_fast

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size + self.text_token_offset

    @property
    def max_token_id(self) -> int:
        return self.tokenizer.max_token_id + self.text_token_offset

    @property
    def max_chars_per_token(self) -> int:
        return self.tokenizer.max_chars_per_token

    @property
    def truncation_side(self) -> str:
        return self.tokenizer.truncation_side

    def __hash__(self) -> int:
        return hash(id(self))

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        encoded = self.tokenizer(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )
        if "input_ids" in encoded:
            encoded["input_ids"] = self._shift_ids_up(encoded["input_ids"])
        return encoded

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()

    def get_added_vocab(self) -> dict[str, int]:
        return self._added_vocab.copy()

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        base_ids = self.tokenizer.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return [tid + self.text_token_offset for tid in base_ids]

    def apply_chat_template(
        self,
        conversation: list["ChatCompletionMessageParam"] | None = None,
        *,
        messages: list["ChatCompletionMessageParam"] | None = None,
        tools: list[dict[str, Any]] | None = None,
        mode: str | None = None,
        task: str | int | None = None,
        **kwargs,
    ) -> str | list[int]:
        """Render a conversation into an OpusLM prompt.

        The rendered text is wrapped in the ESPnet task layout, which is the
        only shape the model was trained on. That has to happen here because a
        request without audio never reaches OpusLMMultiModalProcessor: vLLM's
        input preprocessor returns the tokenized prompt untouched when there is
        no multimodal data, so the tokenizer is the last hook on that path.

        `mode` and `task` select the task token, and are declared explicitly
        rather than swallowed by **kwargs on purpose -- vLLM only forwards
        `chat_template_kwargs` entries that name a real parameter of this
        method (see resolve_chat_template_kwargs in vllm/renderers/hf.py), so a
        catch-all would silently drop them. With neither given, the task
        defaults to plain TTS, matching the model-side resolver.

        Token IDs are returned whether or not `tokenize` was requested. The
        OpenAI chat path asks for a string (preprocess_chat in
        vllm/renderers/online_renderer.py pins `tokenize=False` for every
        non-Mistral tokenizer) and tokenizes it later, and a layout expressed in
        token IDs cannot survive that round trip. Returning IDs is safe because
        parse_dec_only_prompt turns a list of ints into a TokensPrompt, which
        the input preprocessor forwards to the engine unchanged.
        """
        msgs = conversation if conversation is not None else messages
        if msgs is None:
            msgs = []
        out = self.tokenizer.apply_chat_template(msgs, tools=tools, **kwargs)

        if _conversation_has_audio(msgs):
            # Requests carrying audio are laid out by OpusLMMultiModalProcessor,
            # which re-tokenizes the rendered string itself. Adding the layout
            # here would duplicate the markers and hide the audio placeholder.
            return out

        if isinstance(out, list):
            body_ids = [int(tid) + self.text_token_offset for tid in out]
        else:
            body_ids = self.encode(out)

        task_token_id = self._task_layout.resolve_task_token_id(
            has_audio_input=False,
            mode=mode,
            task=task,
        )
        return self._task_layout.apply(body_ids, task_token_id)

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            if tokens in self._special_token_to_id:
                return self._special_token_to_id[tokens]
            return int(self.tokenizer.convert_tokens_to_ids(tokens)) + self.text_token_offset

        out: list[int] = []
        for token in tokens:
            if token in self._special_token_to_id:
                out.append(self._special_token_to_id[token])
            else:
                out.append(
                    int(self.tokenizer.convert_tokens_to_ids(token))
                    + self.text_token_offset
                )
        return out

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, int):
            ids = [ids]
        if not ids:
            return ""

        if skip_special_tokens:
            text_ids = [tid for tid in ids if self._is_text_global_id(int(tid))]
            return self._decode_text_chunk(text_ids, skip_special_tokens=True)

        parts: list[str] = []
        text_chunk: list[int] = []
        for tid in ids:
            tid = int(tid)
            if self._is_text_global_id(tid):
                text_chunk.append(tid)
                continue

            if text_chunk:
                parts.append(
                    self._decode_text_chunk(text_chunk, skip_special_tokens=False)
                )
                text_chunk.clear()

            parts.append(
                self._special_id_to_token.get(tid, f"<|opus_special_{tid}|>")
            )

        if text_chunk:
            parts.append(self._decode_text_chunk(text_chunk, skip_special_tokens=False))

        return "".join(parts)

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        out: list[str] = []
        for tid in ids:
            tid = int(tid)
            if self._is_text_global_id(tid):
                out.extend(
                    self.tokenizer.convert_ids_to_tokens(
                        [self._to_base_id(tid)],
                        skip_special_tokens=skip_special_tokens,
                    )
                )
                continue

            if skip_special_tokens:
                continue
            out.append(self._special_id_to_token.get(tid, f"<|opus_special_{tid}|>"))
        return out

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only runs for attributes missing on self, so delegating
        # "tokenizer" itself would recurse forever. That is reachable: copy
        # and pickle build a bare instance without calling __init__, then
        # probe it (copy._reconstruct does hasattr(obj, "__setstate__")), so
        # self.tokenizer is not set yet. vllm/renderers/hf.py copies the
        # tokenizer on every server start, which hit exactly this.
        if name == "tokenizer":
            raise AttributeError(name)
        return getattr(self.tokenizer, name)
