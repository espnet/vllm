# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from typing import Any

from transformers import BatchEncoding

from vllm.tokenizers.opuslm import OpusLMTokenizer
from vllm.tokenizers.opuslm_dialogue import OpusLMDialogueTokenizer


class _DummyTokenizer:
    def __init__(self) -> None:
        self.all_special_ids = [99]
        self.all_special_tokens = ["<dummy_special>"]
        self.bos_token_id = 11
        self.eos_token_id = 12
        self.pad_token_id = 0
        self.is_fast = True
        self.vocab_size = 100
        self.max_token_id = 99
        self.max_chars_per_token = 8
        self.truncation_side = "left"

    def num_special_tokens_to_add(self) -> int:
        return 0

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        del text, text_pair, add_special_tokens, truncation, max_length, kwargs
        return BatchEncoding(data={"input_ids": [1, 2, 3]})

    def get_vocab(self) -> dict[str, int]:
        return {"a": 1, "b": 2}

    def get_added_vocab(self) -> dict[str, int]:
        return {"<extra>": 77}

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        del text, truncation, max_length, add_special_tokens
        return [10, 20]

    def apply_chat_template(
        self,
        messages,
        tools=None,
        **kwargs,
    ) -> str | list[int]:
        del messages, tools
        if kwargs.get("tokenize", True):
            return [4, 5]
        return "dummy-template"

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            if tokens.startswith("T"):
                return int(tokens[1:])
            return 7
        return [self.convert_tokens_to_ids(t) for t in tokens]  # type: ignore

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def decode(self, ids: list[int] | int, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        return ",".join(str(i) for i in ids)

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        del skip_special_tokens
        return [f"T{i}" for i in ids]


def _make_tokenizer() -> OpusLMTokenizer:
    return OpusLMTokenizer(
        tokenizer=_DummyTokenizer(),  # type: ignore[arg-type]
        text_token_offset=100,
        text_token_end=300,
        pad_token_id=0,
        eos_token_id=5,
        codec_ssl_start_end_token_id=34,
        text_bpe_start_end_token_id=35,
    )


def test_opuslm_tokenizer_shifts_encode_and_call():
    tokenizer = _make_tokenizer()
    assert tokenizer.encode("hello") == [110, 120]

    encoded = tokenizer("hello")
    assert encoded["input_ids"] == [101, 102, 103]


def test_opuslm_tokenizer_decode_skips_non_text_specials():
    tokenizer = _make_tokenizer()
    # text ids are in [100, 300), others are Opus special/modality ids.
    decoded = tokenizer.decode([5, 110, 34, 120, 35], skip_special_tokens=True)
    assert decoded == "10,20"

    tokens = tokenizer.convert_ids_to_tokens(
        [5, 110, 120, 34], skip_special_tokens=True
    )
    assert tokens == ["T10", "T20"]


def test_opuslm_tokenizer_decode_without_skip_includes_markers():
    tokenizer = _make_tokenizer()
    decoded = tokenizer.decode([5, 110, 34, 120, 35], skip_special_tokens=False)
    assert decoded == "<sos/eos>10<codec_ssl_start/end>20<text_bpe_start/end>"


def test_opuslm_tokenizer_apply_chat_template_and_special_ids():
    tokenizer = _make_tokenizer()

    # tokenize=True: the rendered ids are shifted by text_token_offset and then
    # wrapped in the ESPnet task layout. With no mode given the task defaults to
    # plain TTS (82), matching the model-side resolver.
    tokenized = tokenizer.apply_chat_template([], tokenize=True)
    assert tokenized == [5, 82, 35, 104, 105] + [0] * 8 + [34]

    assert tokenizer.convert_tokens_to_ids("<codec_ssl_start/end>") == 34
    assert tokenizer.convert_tokens_to_ids("T10") == 110


def test_opuslm_tokenizer_lays_out_even_when_asked_for_a_string():
    """tokenize=False must still come back laid out, as token IDs.

    preprocess_chat in vllm/renderers/online_renderer.py pins tokenize=False for
    every non-Mistral tokenizer, renders to a string and tokenizes it later. A
    layout expressed in token IDs cannot survive that round trip: the server
    would send the model a bare text prompt with no <sos/eos>, no task token and
    no segment markers. Returning IDs is what parse_dec_only_prompt needs to
    build a TokensPrompt, which the input preprocessor forwards unchanged.
    """
    tokenizer = _make_tokenizer()

    # The dummy renders "dummy-template", which the dummy encoder maps to
    # [10, 20] and the wrapper shifts to [110, 120].
    assert tokenizer.apply_chat_template([], tokenize=False) == (
        [5, 82, 35, 110, 120] + [0] * 8 + [34]
    )
    assert tokenizer.apply_chat_template([], mode="text_text", tokenize=False) == (
        [5, 64, 110, 120]
    )


def test_opuslm_tokenizer_leaves_audio_requests_to_the_mm_processor():
    """A conversation carrying audio must come back untouched.

    OpusLMMultiModalProcessor re-tokenizes the rendered string and lays out the
    audio prompt itself (_layout_task_sequence). Laying it out here as well would
    duplicate the markers and hide the audio placeholder from the processor.
    """
    tokenizer = _make_tokenizer()

    # "openai" content format: the audio part survives as a dict.
    parts = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]
    assert tokenizer.apply_chat_template(parts, tokenize=False) == "dummy-template"

    # "string" content format: the part is already the model's placeholder.
    placeholder = [{"role": "user", "content": "<codec_ssl_start_end>"}]
    assert tokenizer.apply_chat_template(placeholder, tokenize=False) == "dummy-template"

    # Text-only messages in the same shapes still get laid out.
    text_parts = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert tokenizer.apply_chat_template(text_parts, mode="text_text", tokenize=False) == (
        [5, 64, 110, 120]
    )


def test_opuslm_tokenizer_apply_chat_template_lays_out_by_mode():
    """Text-only requests get their task layout here and nowhere else.

    vllm/inputs/preprocess.py returns a tokenized prompt untouched when the
    request carries no multimodal data, so OpusLMMultiModalProcessor -- which
    lays out prompts that do carry audio -- never runs for these. Without this
    layout the model sees a bare text prompt with no <sos/eos>, no task token
    and no segment markers, and generates from conditioning it never saw in
    training.
    """
    tokenizer = _make_tokenizer()
    body = [104, 105]

    # text_audio -> plain TTS (82): text condition segment, then nq-1 pad frames
    # so ARDelay interleaving cannot overlap the condition and the target, then
    # the codec target prefix.
    assert tokenizer.apply_chat_template([], mode="text_audio", tokenize=True) == (
        [5, 82, 35] + body + [0] * 8 + [34]
    )

    # text_text -> textlm (64): no audio anywhere, so no markers and no pads.
    assert tokenizer.apply_chat_template([], mode="text_text", tokenize=True) == (
        [5, 64] + body
    )

    # An explicit task overrides mode, and accepts both names and raw ids.
    assert tokenizer.apply_chat_template([], task="textlm", tokenize=True) == (
        [5, 64] + body
    )
    assert tokenizer.apply_chat_template(
        [], mode="text_text", task="plain_tts", tokenize=True
    ) == ([5, 82, 35] + body + [0] * 8 + [34])
    assert tokenizer.apply_chat_template([], task=83, tokenize=True) == [5, 83] + body


def test_opuslm_tokenizer_apply_chat_template_declares_mode_explicitly():
    """`mode` and `task` must be real parameters, not swallowed by **kwargs.

    resolve_chat_template_kwargs in vllm/renderers/hf.py forwards only the
    chat_template_kwargs entries that name a declared parameter of
    apply_chat_template, and silently drops the rest. A catch-all **kwargs
    would therefore never receive `mode`, and every request would fall back to
    the default task with no error anywhere.
    """
    import inspect

    params = inspect.signature(OpusLMTokenizer.apply_chat_template).parameters
    assert "mode" in params
    assert "task" in params
    assert params["mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["task"].kind is inspect.Parameter.KEYWORD_ONLY


def test_opuslm_tokenizer_survives_copy():
    # vllm/renderers/hf.py does copy.copy(tokenizer) on every server start.
    # copy builds a bare instance without running __init__ and then probes it
    # for __setstate__, so a __getattr__ that delegates "tokenizer" to
    # self.tokenizer recurses until RecursionError.
    tokenizer = _make_tokenizer()

    shallow = copy.copy(tokenizer)
    assert shallow.encode("hello") == [110, 120]

    deep = copy.deepcopy(tokenizer)
    assert deep.encode("hello") == [110, 120]


def _make_dialogue_tokenizer() -> OpusLMDialogueTokenizer:
    return OpusLMDialogueTokenizer(
        tokenizer=_DummyTokenizer(),  # type: ignore[arg-type]
        text_token_offset=100,
        text_token_end=300,
        pad_token_id=0,
        eos_token_id=5,
        codec_ssl_start_end_token_id=34,
        text_bpe_start_end_token_id=35,
    )


def test_opuslm_dialogue_tokenizer_survives_copy():
    tokenizer = _make_dialogue_tokenizer()

    assert copy.copy(tokenizer).encode("hello") == [110, 120]
    assert copy.deepcopy(tokenizer).encode("hello") == [110, 120]


def test_opuslm_dialogue_tokenizer_lays_out_a_text_dialogue():
    """A dialogue with no audio gets its turn layout here and nowhere else.

    _build_dialogue_sequence in the model lays out dialogues that carry audio,
    but it only runs when the request has multimodal data. A text dialogue that
    skipped this would reach the model as bare BPE: no <sos/eos>, no task token,
    no role or modality markers, and no generation target for it to continue
    from.
    """
    tokenizer = _make_dialogue_tokenizer()
    body = [110, 120]  # the dummy encoder's ids, shifted by text_token_offset

    # One user turn, text out: <sos/eos> <text_dialogue> <user> <text_bpe> body
    # + nq-1 pads, then the two tokens that mark the target segment.
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False
    ) == ([5, 88, 9, 35] + body + [0] * 8 + [10, 35])

    # Multi-part text content reaches the same layout.
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], tokenize=False
    ) == ([5, 88, 9, 35] + body + [0] * 8 + [10, 35])

    # The client marks the generation target with an empty assistant message.
    # apply_dialogue appends that marker itself, so the empty turn must not
    # become a segment of its own.
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}],
        tokenize=False,
    ) == ([5, 88, 9, 35] + body + [0] * 8 + [10, 35])

    # A system turn keeps its own role token.
    assert tokenizer.apply_chat_template(
        [{"role": "system", "content": "hi"}, {"role": "user", "content": "hi"}],
        mode="text_dialogue",
        tokenize=False,
    ) == ([5, 88, 8, 35] + body + [0] * 8 + [9, 35] + body + [0] * 8 + [10, 35])


def test_opuslm_dialogue_target_modality_follows_the_task_and_last_turn():
    """An audio dialogue speaks only after it has written the reply as text.

    ESPnet splits an audio dialogue turn in two: the assistant emits the reply
    as text, then the same reply as codec frames. So the target is codec exactly
    when the turn before it is the assistant's own text. A text dialogue never
    targets codec whatever the history looks like.
    """
    tokenizer = _make_dialogue_tokenizer()
    body = [110, 120]

    # Audio dialogue, assistant has not written yet -> the target is its text.
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], mode="audio_dialogue", tokenize=False
    ) == ([5, 89, 9, 35] + body + [0] * 8 + [10, 35])

    # Audio dialogue, assistant text already there -> the target is codec.
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        mode="audio_dialogue",
        tokenize=False,
    ) == ([5, 89, 9, 35] + body + [0] * 8 + [10, 35] + body + [0] * 8 + [10, 34])

    # The same history under the text task still targets text.
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        mode="text_dialogue",
        tokenize=False,
    ) == ([5, 88, 9, 35] + body + [0] * 8 + [10, 35] + body + [0] * 8 + [10, 35])


def test_opuslm_dialogue_tokenizer_leaves_audio_requests_to_the_mm_processor():
    """A dialogue carrying audio must come back as the rendered string.

    _build_dialogue_sequence re-tokenizes that string and lays out the speaker
    prompt, the audio placeholders and the stream replay itself. Laying the turns
    out here as well would duplicate every marker and hide the placeholders the
    processor looks for.
    """
    tokenizer = _make_dialogue_tokenizer()

    parts = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]
    assert tokenizer.apply_chat_template(parts, tokenize=False) == "dummy-template"

    placeholder = [{"role": "user", "content": "<codec_ssl_start_end>"}]
    assert tokenizer.apply_chat_template(placeholder, tokenize=False) == "dummy-template"


def test_opuslm_dialogue_tokenizer_declares_mode_explicitly():
    """`mode` and `task` must be real parameters, not swallowed by **kwargs.

    resolve_chat_template_kwargs in vllm/renderers/hf.py forwards only the
    chat_template_kwargs entries naming a declared parameter, and drops the rest
    silently -- a catch-all would leave every dialogue on the default task with
    no error anywhere.
    """
    import inspect

    params = inspect.signature(OpusLMDialogueTokenizer.apply_chat_template).parameters
    assert params["mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["task"].kind is inspect.Parameter.KEYWORD_ONLY


def test_opuslm_tokenizer_kwargs_are_read_off_the_config():
    """Both tokenizers must be configured from the checkpoint's own config.

    Their fallbacks are the released checkpoints' numbering, and the layout they
    build is not validated against anything downstream, so a checkpoint that
    renumbers a task token or a role marker and is not read here would be laid
    out with the wrong ids and no error anywhere.
    """
    from types import SimpleNamespace

    from vllm.tokenizers.registry import _inject_opuslm_tokenizer_kwargs

    # Every value here differs from the released checkpoints', so any id that
    # comes back at its fallback was not read off the config.
    cfg = SimpleNamespace(
        text_token_start=7000,
        text_token_end=40000,
        pad_token_id=0,
        eos_token_id=6,
        codec_ssl_start_end_token_id=44,
        text_bpe_start_end_token_id=45,
        textlm_task_token_id=164,
        codec_ssl_asr_task_token_id=180,
        codec_ssl_tts_task_token_id=181,
        codec_ssl_plain_tts_task_token_id=182,
        codec_ssl_audiolm_task_token_id=183,
        text_dialogue_task_token_id=188,
        audio_dialogue_task_token_id=189,
        nq=5,
        system_prompt_token_id=18,
        user_input_token_id=19,
        assistant_output_token_id=20,
    )

    base = _inject_opuslm_tokenizer_kwargs("opuslm", cfg, {})
    assert base["opuslm_text_token_offset"] == 7000
    assert base["opuslm_text_token_end"] == 40000
    assert base["opuslm_codec_ssl_plain_tts_task_token_id"] == 182
    assert base["opuslm_nq"] == 5

    dialogue = _inject_opuslm_tokenizer_kwargs("opuslm_dialogue", cfg, {})
    assert dialogue["opuslm_dialogue_text_token_offset"] == 7000
    assert dialogue["opuslm_dialogue_text_dialogue_task_token_id"] == 188
    # The role markers are what a dialogue layout is built out of, so the
    # dialogue tokenizer needs three ids the base one never asks for.
    assert dialogue["opuslm_dialogue_system_prompt_token_id"] == 18
    assert dialogue["opuslm_dialogue_user_input_token_id"] == 19
    assert dialogue["opuslm_dialogue_assistant_output_token_id"] == 20
    assert "opuslm_user_input_token_id" not in base

    # A kwarg the caller passed explicitly still wins over the config.
    pinned = _inject_opuslm_tokenizer_kwargs("opuslm", cfg, {"opuslm_nq": 9})
    assert pinned["opuslm_nq"] == 9
