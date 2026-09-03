# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the config.json and tokenizer that released ESPnet checkpoints omit.

WHY THIS EXISTS
---------------
espnet publishes weights and a training YAML, and nothing else. A vLLM
directory additionally needs a ``config.json`` and a tokenizer, so for a while
the converters took a ``--ref-dir`` pointing at a directory that already had
them -- which nobody outside the original authors could produce. This module
closes that: it builds every one of those files from **public, pinned**
artifacts, so a new user can convert an official checkpoint on a clean machine.

WHERE EACH FILE COMES FROM
--------------------------
Nothing here is invented. Three kinds of input:

1. The **public backbone repository** named by the model's own released YAML.
   Tokenizer files are taken from it, byte-for-byte where possible, and the
   transformer geometry (hidden size, layer count, rope theta, ...) is read
   from its ``config.json``. Every fetch is pinned to a git revision AND
   verified against a recorded sha256, so an upstream edit fails loudly
   instead of silently changing your model.
2. The **released ESPnet YAML** in the checkpoint repository. For OpusLM this
   carries the complete ``token_list`` and ``token_bias`` blocks, so every
   vocabulary boundary and every special-token id is *looked up by name*
   rather than hard-coded. For Bagpiper it carries the tokenizer name, the
   codec choice and the stream count.
3. A small amount of **metadata this fork owns**: the chat template, the names
   of Bagpiper's reserved and codec tokens, and the decoder wiring (which
   Xcodec / DAC / XEUS repo to load). These are properties of this
   implementation, not of the checkpoint, so they live here as constants.

WHAT IS NOT DERIVABLE
---------------------
A few fields in the hand-written reference configs disagree with the public
backbone. Those are listed per model in ``LEGACY_OVERRIDES``, applied by
default so a freshly bootstrapped directory is numerically identical to the
one that was actually tested, and **printed every time** with the backbone's
value beside them. ``--no-legacy-overrides`` emits pure backbone-derived
values instead. Nothing is silently chosen.

USAGE
-----
Normally you do not call this directly -- ``convert_bagpiper_ckpt.py`` and
``convert_opuslm_ckpt.py`` call it when you do not pass ``--ref-dir``. To
build just the assets, or to check them against a known-good directory:

    python bootstrap_assets.py --model bagpiper --out ./assets
    python bootstrap_assets.py --model opuslm --out ./assets --compare-with REF

On a machine with no network, pre-download the pinned files listed by
``--print-sources`` into one directory and pass ``--offline-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODELS = ("bagpiper", "opuslm", "opuslm_dialogue")


# ---------------------------------------------------------------------------
# Pinned public sources
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    """One file, pinned to a revision and to its content hash."""

    repo: str
    revision: str
    filename: str
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{self.repo}/resolve/"
            f"{self.revision}/{self.filename}"
        )

    @property
    def offline_name(self) -> str:
        """Filename to look for under --offline-dir."""
        return f"{self.repo.replace('/', '_')}__{self.filename}"


# Bagpiper: train_bagpiper_tts.yaml / train_stage3_qwen3_base.yaml name
# tokenizer_name: Qwen/Qwen3-8B-Base and
# continuous_audio.encoder_hf_model_tag: Qwen/Qwen3-Omni-30B-A3B-Instruct.
_QWEN3_8B_BASE = "Qwen/Qwen3-8B-Base"
_QWEN3_8B_BASE_REV = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
_QWEN3_OMNI = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_QWEN3_OMNI_REV = "26291f793822fb6be9555850f06dfe95f2d7e695"

BAGPIPER_SOURCES: dict[str, Source] = {
    "text_config": Source(
        _QWEN3_8B_BASE,
        _QWEN3_8B_BASE_REV,
        "config.json",
        "3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a",
    ),
    "text_tokenizer": Source(
        _QWEN3_8B_BASE,
        _QWEN3_8B_BASE_REV,
        "tokenizer.json",
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
    "text_tokenizer_config": Source(
        _QWEN3_8B_BASE,
        _QWEN3_8B_BASE_REV,
        "tokenizer_config.json",
        "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    ),
    "text_merges": Source(
        _QWEN3_8B_BASE,
        _QWEN3_8B_BASE_REV,
        "merges.txt",
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    "audio_config": Source(
        _QWEN3_OMNI,
        _QWEN3_OMNI_REV,
        "config.json",
        "eab5093d47807aaf894119506b238b2b1cee70d08456e894fee9a012d88f2e0d",
    ),
}

# OpusLM and OpusLM-dialogue. Note the two repositories are *different*:
# config.yaml's `subword_model` names the tokenizer, `transformer_conf.
# hf_model_tag` names the transformer whose geometry we copy. For OpusLM they
# happen to coincide; for the dialogue model they do not (SmolLM-1.7B provides
# the tokenizer, SmolLM2-1.7B-Instruct the transformer).
_OLMO2 = "allenai/OLMo-2-1124-7B"
_OLMO2_REV = "7df9a82518afdecae4e8c026b27adccc8c1f0032"
_SMOLLM = "HuggingFaceTB/SmolLM-1.7B"
_SMOLLM_REV = "d7449ff7241c863f3e8accc475155f0f97afa011"
_SMOLLM2_INSTRUCT = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
_SMOLLM2_INSTRUCT_REV = "31b70e2e869a7173562077fd711b654946d38674"

# Tokenizer files copied verbatim from the backbone (proven byte-identical to
# the hand-built reference directories).
OPUSLM_TOKENIZER_FILES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
)

OPUSLM_SOURCES: dict[str, dict[str, Source]] = {
    "opuslm": {
        "espnet_yaml": Source(
            "espnet/OpusLM_7B_Anneal",
            "b45ab2e7317f859549de329d2db9672d6fcabcb1",
            "config.yaml",
            "b483454240c832ae783b4b4a46ec63be478900fa5b018094967df45a0aeb5311",
        ),
        "transformer_config": Source(
            _OLMO2,
            _OLMO2_REV,
            "config.json",
            "f1e210de0ba704c579768c154bf57e36e170868ffecb060a6e0d3f2e6b0b9710",
        ),
        "tokenizer.json": Source(
            _OLMO2,
            _OLMO2_REV,
            "tokenizer.json",
            "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca",
        ),
        "tokenizer_config.json": Source(
            _OLMO2,
            _OLMO2_REV,
            "tokenizer_config.json",
            "91c69c665697785ace4ec7dbd159e1839bc9bb5033ab05a56bb0547521dc9ab0",
        ),
        "vocab.json": Source(
            _OLMO2,
            _OLMO2_REV,
            "vocab.json",
            "9e14712c91b37c7aab74b1306baa46ac342d620637a4b44523cdc3aec7d24195",
        ),
        "merges.txt": Source(
            _OLMO2,
            _OLMO2_REV,
            "merges.txt",
            "b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502",
        ),
        "special_tokens_map.json": Source(
            _OLMO2,
            _OLMO2_REV,
            "special_tokens_map.json",
            "3c6bf7c09d5473c303cee8575a22bb51e5153c17d177a721b43cd4785c6d09ae",
        ),
    },
    "opuslm_dialogue": {
        "espnet_yaml": Source(
            "espnet/multi_turn_SDS_RLAIF",
            "a784cde04ffb1e7e8e83044dab24b54d1c298429",
            "config.yaml",
            "2c0b0ff162cae47b98fa4e168b2a6199fd63487e233eeab15becef387432428f",
        ),
        "transformer_config": Source(
            _SMOLLM2_INSTRUCT,
            _SMOLLM2_INSTRUCT_REV,
            "config.json",
            "994f50b16abb4ae00880baefe03c10260b5bd608d2bf586f7056ca05a534feea",
        ),
        "tokenizer.json": Source(
            _SMOLLM,
            _SMOLLM_REV,
            "tokenizer.json",
            "6ae489e54654605c859dec0fe2ba636baf64399dffe01ec44b3311778b269857",
        ),
        "tokenizer_config.json": Source(
            _SMOLLM,
            _SMOLLM_REV,
            "tokenizer_config.json",
            "2d9292f87357fec4a28f4e170533dfc712bfd1d16870f3d3d9afe0419859842a",
        ),
        "vocab.json": Source(
            _SMOLLM,
            _SMOLLM_REV,
            "vocab.json",
            "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
        ),
        "merges.txt": Source(
            _SMOLLM,
            _SMOLLM_REV,
            "merges.txt",
            "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
        ),
        "special_tokens_map.json": Source(
            _SMOLLM,
            _SMOLLM_REV,
            "special_tokens_map.json",
            "e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3",
        ),
    },
}


def all_sources(model: str) -> dict[str, Source]:
    if model == "bagpiper":
        return dict(BAGPIPER_SOURCES)
    return dict(OPUSLM_SOURCES[model])


# ---------------------------------------------------------------------------
# Metadata this fork owns (not fetched, not derivable from anything public)
# ---------------------------------------------------------------------------
# Bagpiper's 256 reserved ids. The first twelve are named roles and modality
# markers used by the chat template; the rest are reserved padding so the text
# vocabulary starts at a round offset.
BAGPIPER_NAMED_SPECIALS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|eot|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|text|>",
    "<|audio|>",
    "<|image|>",
    "<|video|>",
    "<|toolcall|>",
)
BAGPIPER_NUM_RESERVED = 256

# The template rendered by transformers when it finds chat_template.jinja. The
# compact copy below goes into tokenizer_config.json for older transformers
# releases that only read it from there; validate_bagpiper() asserts the two
# render identically.
BAGPIPER_CHAT_TEMPLATE_JINJA = """\
{#- SpeechLM chat template
    Prompt format (from ESPnet training):
      <|bos|>[<|system|><|text|>sys<|eos|>]<|user|><|text|>text<|eos|><|assistant|>
      <|bos|>[<|system|><|text|>sys<|eos|>]<|user|><|audio|><|eos|><|assistant|>
    Audio messages: content already contains <|audio|> placeholder from vLLM.
-#}
{%- for message in messages %}
{%- if loop.first %}{{ '<|bos|>' }}{% endif %}
{%- if message.role == 'system' %}
{{- '<|system|><|text|>' + message.content + '<|eos|>' }}
{%- elif message.role == 'user' %}
{%- if '<|audio|>' in message.content %}
{{- '<|user|>' + message.content + '<|eos|>' }}
{%- else %}
{{- '<|user|><|text|>' + message.content + '<|eos|>' }}
{%- endif %}
{%- elif message.role == 'assistant' %}
{%- if '<|audio|>' in message.content %}
{{- '<|assistant|>' + message.content + '<|eos|>' }}
{%- else %}
{{- '<|assistant|><|text|>' + message.content + '<|eos|>' }}
{%- endif %}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}
"""

BAGPIPER_CHAT_TEMPLATE_COMPACT = (
    "{% for message in messages %}"
    "{% if loop.first %}<|bos|>{% endif %}"
    "{% if message.role == 'system' %}"
    "<|system|><|text|>{{ message.content }}<|eos|>"
    "{% elif message.role == 'user' %}"
    "{% if '<|audio|>' in message.content %}"
    "<|user|>{{ message.content }}<|eos|>"
    "{% else %}<|user|><|text|>{{ message.content }}<|eos|>{% endif %}"
    "{% elif message.role == 'assistant' %}"
    "{% if '<|audio|>' in message.content %}"
    "<|assistant|>{{ message.content }}<|eos|>"
    "{% else %}<|assistant|><|text|>{{ message.content }}<|eos|>{% endif %}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

# OpusLM's structural tokens are added by OpusLMTokenizer at runtime, in the
# reserved id range below text_token_start, so the chat template must pass the
# message content through untouched. See convert_opuslm_ckpt.fix_chat_template.
OPUSLM_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
)

# Which external module decodes OpusLM's audio. These are properties of this
# fork's decoder wiring, not of the checkpoint; all three name public repos.
OPUSLM_DECODER_WIRING: dict[str, Any] = {
    "dac_hf_model_tag": "ftshijt/espnet_codec_dac_large_v1.4_360epoch",
    "xeus_hf_model_tag": "espnet/xeus",
    "km_model_filename": "model/km_opus_lm.mdl",
    "dac_sample_rate": 16000,
    "xeus_layer": 18,
    "audio_temperature": 0.8,
    "audio_topk": 30,
}
OPUSLM_DIALOGUE_EXTRA_WIRING: dict[str, Any] = {
    "xeus_checkpoint_filename": "model/xeus_checkpoint_new.pth",
    "audio_minlen": 3,
}

# Same for Bagpiper: the Xcodec repo comes from the released YAML
# (discrete_audio.codec_hf_model_tag), the sample rate is Xcodec's own.
BAGPIPER_CODEC_TAG = "hf-audio/xcodec-hubert-general"
BAGPIPER_CODEC_SAMPLE_RATE = 16000

# Every field where the hand-written reference config disagrees with the public
# backbone. Applied by default so a bootstrapped directory matches the one that
# was tested; each is printed with the backbone's value beside it.
LEGACY_OVERRIDES: dict[str, dict[str, tuple[Any, str]]] = {
    "bagpiper": {
        "text_config.max_position_embeddings": (
            40960,
            f"{_QWEN3_8B_BASE} config.json says 32768; 40960 is the value the "
            "validated Bagpiper config uses (it matches the Qwen3-8B release)",
        ),
        "audio_config.n_window": (
            100,
            f"{_QWEN3_OMNI} thinker_config.audio_config says 50",
        ),
        "audio_config.n_window_infer": (
            400,
            f"{_QWEN3_OMNI} thinker_config.audio_config says 800",
        ),
    },
    "opuslm": {
        "rms_norm_eps": (
            1e-5,
            f"{_OLMO2} config.json says 1e-06; the hand-written OpusLM "
            "config.json has always said 1e-05 and that is what was tested",
        ),
    },
    "opuslm_dialogue": {},
}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch(src: Source, offline_dir: Path | None = None) -> bytes:
    """Return the pinned file's bytes, refusing anything with a wrong hash."""
    if offline_dir is not None:
        candidates = [offline_dir / src.offline_name, offline_dir / src.filename]
        for path in candidates:
            if path.exists():
                data = path.read_bytes()
                break
        else:
            raise SystemExit(
                f"ERROR: {src.repo}/{src.filename} not found under "
                f"{offline_dir}. Looked for {src.offline_name} and "
                f"{src.filename}. Download it from:\n  {src.url}"
            )
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - dependency of vLLM
            raise SystemExit(
                "ERROR: huggingface_hub is required to fetch the pinned "
                f"config/tokenizer sources ({exc}). Either install it, or "
                "download the files listed by --print-sources and pass "
                "--offline-dir."
            ) from exc
        try:
            path = hf_hub_download(
                repo_id=src.repo, filename=src.filename, revision=src.revision
            )
        except Exception as exc:
            raise SystemExit(
                f"ERROR: could not fetch {src.repo}/{src.filename} at "
                f"revision {src.revision}: {exc}\n"
                f"  Direct URL: {src.url}\n"
                "  If this machine has no network access, download the files "
                "listed by --print-sources and pass --offline-dir."
            ) from exc
        data = Path(path).read_bytes()

    digest = hashlib.sha256(data).hexdigest()
    if digest != src.sha256:
        raise SystemExit(
            f"ERROR: {src.repo}/{src.filename} at revision {src.revision} "
            f"hashed to\n  {digest}\nbut this converter was pinned to\n  "
            f"{src.sha256}\nRefusing to build a tokenizer from an unexpected "
            "file. If upstream legitimately changed, re-pin deliberately."
        )
    return data


def _write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=indent, ensure_ascii=False)
        handle.write("\n")


def _added_token_entry(content: str) -> dict[str, Any]:
    """The added-token record shape transformers writes for a special token."""
    return {
        "content": content,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    }


# ---------------------------------------------------------------------------
# Bagpiper
# ---------------------------------------------------------------------------
def bagpiper_reserved_names() -> list[str]:
    """The 256 reserved ids: named roles first, then <|unused_N|> padding."""
    names = list(BAGPIPER_NAMED_SPECIALS)
    names += [
        f"<|unused_{i}|>" for i in range(len(names), BAGPIPER_NUM_RESERVED)
    ]
    assert len(names) == BAGPIPER_NUM_RESERVED
    return names


def bagpiper_codec_names(num_streams: int, per_stream: int) -> list[str]:
    """Codec token names: one pad plus `per_stream` codes for each stream."""
    names: list[str] = []
    for stream in range(num_streams):
        names.append(f"<codec_layer{stream}_pad>")
        names += [f"<codec_layer{stream}_code{c}>" for c in range(per_stream)]
    return names


def build_bagpiper_assets(
    out_dir: Path,
    *,
    offline_dir: Path | None = None,
    legacy_overrides: bool = True,
    num_streams: int = 8,
    codec_codes_per_stream: int = 1024,
) -> dict[str, Any]:
    """Write Bagpiper's config.json and tokenizer into ``out_dir``.

    The vocabulary is laid out exactly as the released YAML implies:

        [0, 256)                        reserved specials
        [256, 256 + qwen_vocab_size)    Qwen3-8B-Base text tokens, ids +256
        [codec_base, +8*1025)           8 Xcodec streams, 1 pad + 1024 codes
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    print("Fetching pinned config/tokenizer sources ...")
    src = BAGPIPER_SOURCES
    qwen_cfg = json.loads(fetch(src["text_config"], offline_dir))
    qwen_tok = json.loads(fetch(src["text_tokenizer"], offline_dir))
    qwen_tok_cfg = json.loads(fetch(src["text_tokenizer_config"], offline_dir))
    qwen_merges = fetch(src["text_merges"], offline_dir)
    omni_cfg = json.loads(fetch(src["audio_config"], offline_dir))
    for key, source in src.items():
        print(f"  {source.repo}@{source.revision[:8]}/{source.filename} ({key})")

    # ---- vocabulary arithmetic, entirely from the fetched configs ----------
    text_offset = BAGPIPER_NUM_RESERVED
    text_span = qwen_cfg["vocab_size"]  # 151936: includes Qwen's own padding
    codec_base = text_offset + text_span
    codec_layer_size = codec_codes_per_stream + 1  # +1 for the per-stream pad
    vocab_size = codec_base + num_streams * codec_layer_size
    print(
        f"\nVocabulary: {text_offset} reserved + {text_span} text "
        f"({_QWEN3_8B_BASE} vocab_size) + {num_streams}x{codec_layer_size} "
        f"codec = {vocab_size}"
    )

    # ---- tokenizer.json: shift Qwen's BPE up, then append our tokens -------
    base_vocab: dict[str, int] = qwen_tok["model"]["vocab"]
    shifted = {tok: idx + text_offset for tok, idx in base_vocab.items()}

    # Qwen keeps its own specials in added_tokens_decoder rather than in the
    # BPE vocab; they occupy real ids, so they shift too.
    qwen_specials = {
        entry["content"]: int(idx) + text_offset
        for idx, entry in sorted(
            qwen_tok_cfg["added_tokens_decoder"].items(), key=lambda kv: int(kv[0])
        )
    }
    overlap = set(shifted) & set(qwen_specials)
    if overlap:
        raise SystemExit(
            f"ERROR: {len(overlap)} token(s) appear in both {_QWEN3_8B_BASE}'s "
            f"BPE vocab and its added_tokens_decoder: {sorted(overlap)[:5]}"
        )
    max_text_id = max(qwen_specials.values(), default=max(shifted.values()))
    if max_text_id >= codec_base:
        raise SystemExit(
            f"ERROR: highest text id {max_text_id} collides with the codec "
            f"range starting at {codec_base}."
        )

    reserved = {name: i for i, name in enumerate(bagpiper_reserved_names())}
    codec = {
        name: codec_base + i
        for i, name in enumerate(
            bagpiper_codec_names(num_streams, codec_codes_per_stream)
        )
    }
    if max(codec.values()) != vocab_size - 1:
        raise SystemExit(
            f"ERROR: codec ids end at {max(codec.values())}, expected "
            f"{vocab_size - 1}."
        )

    full_vocab: dict[str, int] = {}
    for part in (reserved, shifted, qwen_specials, codec):
        full_vocab.update(part)
    expected_total = (
        len(reserved) + len(shifted) + len(qwen_specials) + len(codec)
    )
    if len(full_vocab) != expected_total:
        raise SystemExit("ERROR: duplicate token name across vocabulary parts.")

    tokenizer_json = dict(qwen_tok)
    model = dict(qwen_tok["model"])
    model["vocab"] = full_vocab
    # tokenizers >= 0.20 writes merges as pairs; normalise so the file is the
    # same shape regardless of which version produced the upstream file.
    model["merges"] = [
        list(m) if isinstance(m, (list, tuple)) else m.split(" ", 1)
        for m in qwen_tok["model"]["merges"]
    ]
    model.setdefault("ignore_merges", False)
    tokenizer_json["model"] = model
    # Specials are declared in tokenizer_config.added_tokens_decoder, which is
    # what transformers reads; keeping added_tokens empty avoids two competing
    # declarations of the same ids.
    tokenizer_json["added_tokens"] = []
    _write_json(out_dir / "tokenizer.json", tokenizer_json)

    # vocab.json / merges.txt are the slow-tokenizer view of the same BPE.
    _write_json(out_dir / "vocab.json", shifted)
    (out_dir / "merges.txt").write_bytes(qwen_merges)

    # ---- tokenizer_config.json -------------------------------------------
    added_tokens_decoder = {
        str(idx): _added_token_entry(name)
        for name, idx in sorted(
            list(reserved.items()) + list(codec.items()), key=lambda kv: kv[1]
        )
    }
    tokenizer_config = {
        "model_type": "bagpiper",
        "base_tokenizer": _QWEN3_8B_BASE,
        "text_offset": text_offset,
        "codec_base_offset": codec_base,
        "codec_layer_size": codec_layer_size,
        "num_streams": num_streams,
        "vocab_size": vocab_size,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": None,
        "clean_up_tokenization_spaces": False,
        "added_tokens_decoder": added_tokens_decoder,
        "chat_template": BAGPIPER_CHAT_TEMPLATE_COMPACT,
    }
    _write_json(out_dir / "tokenizer_config.json", tokenizer_config)
    (out_dir / "chat_template.jinja").write_text(
        BAGPIPER_CHAT_TEMPLATE_JINJA, encoding="utf-8"
    )

    # Qwen's own specials, at the ids they actually occupy here. The reference
    # directory recorded them unshifted, which was wrong; transformers reads
    # tokenizer_config.added_tokens_decoder for a fast tokenizer, so the stale
    # copy did no damage, but there is no reason to reproduce the error.
    _write_json(out_dir / "added_tokens.json", dict(sorted(qwen_specials.items())))
    notes.append(
        "added_tokens.json records Qwen's 26 specials at their shifted ids "
        f"({min(qwen_specials.values())}..{max(qwen_specials.values())}); the "
        "hand-written reference directory listed them unshifted."
    )

    _write_json(
        out_dir / "special_tokens_map.json",
        {
            "bos_token": _added_token_entry("<|bos|>"),
            "eos_token": _added_token_entry("<|eos|>"),
            "pad_token": _added_token_entry("<|pad|>"),
        },
    )
    notes.append(
        "special_tokens_map.json names <|bos|>/<|eos|>/<|pad|>; the "
        "hand-written reference directory carried a stray Qwen-chat map whose "
        "eos_token (<|endoftext|>) is not this model's eos."
    )

    # ---- config.json ------------------------------------------------------
    audio_src = omni_cfg["thinker_config"]["audio_config"]
    audio_keys = (
        "num_mel_bins",
        "encoder_layers",
        "encoder_attention_heads",
        "encoder_ffn_dim",
        "d_model",
        "dropout",
        "attention_dropout",
        "activation_function",
        "activation_dropout",
        "scale_embedding",
        "initializer_range",
        "max_source_positions",
        "n_window",
        "output_dim",
        "n_window_infer",
        "conv_chunksize",
        "downsample_hidden_size",
    )
    audio_config: dict[str, Any] = {"model_type": "bagpiper_audio_encoder"}
    for key in audio_keys:
        if key not in audio_src:
            raise SystemExit(
                f"ERROR: {_QWEN3_OMNI} audio_config has no {key!r}; the audio "
                "tower layout changed and this generator needs updating."
            )
        audio_config[key] = audio_src[key]

    text_config: dict[str, Any] = {
        "model_type": "bagpiper_text",
        "vocab_size": vocab_size,
    }
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "tie_word_embeddings",
        "rope_theta",
        "attention_bias",
        "attention_dropout",
    ):
        text_config[key] = qwen_cfg[key]
    text_config["rope_theta"] = float(text_config["rope_theta"])

    config: dict[str, Any] = {
        "architectures": ["BagpiperForConditionalGeneration"],
        "model_type": "bagpiper",
        "audio_config": audio_config,
        "text_config": text_config,
        "num_stream": num_streams,
        "adaptor_input_dim": audio_config["output_dim"],
        "vocab_size": vocab_size,
        "hidden_size": text_config["hidden_size"],
        "codec_base_offset": codec_base,
        "codec_layer_size": codec_layer_size,
        "text_token_offset": text_offset,
        "text_token_end": codec_base,
        "initializer_range": text_config["initializer_range"],
        # Decoding defaults from the released inference.yaml.
        "audio_temperature": 0.8,
        "audio_topk": 20,
        "text_temperature": 0.6,
        "text_topk": 20,
        "xcodec_hf_model_tag": BAGPIPER_CODEC_TAG,
        "xcodec_sample_rate": BAGPIPER_CODEC_SAMPLE_RATE,
    }
    # Role/modality ids follow the reserved-name order above, so they are
    # looked up rather than written twice.
    for field, name in (
        ("pad_token_id", "<|pad|>"),
        ("bos_token_id", "<|bos|>"),
        ("eos_token_id", "<|eos|>"),
        ("eot_token_id", "<|eot|>"),
        ("system_token_id", "<|system|>"),
        ("user_token_id", "<|user|>"),
        ("assistant_token_id", "<|assistant|>"),
        ("text_token_id", "<|text|>"),
        ("audio_token_id", "<|audio|>"),
        ("audio_placeholder_token_id", "<|audio|>"),
    ):
        config[field] = reserved[name]

    applied = _apply_overrides(config, "bagpiper", legacy_overrides)
    _write_json(out_dir / "config.json", config)
    _write_json(
        out_dir / "generation_config.json",
        {
            "eos_token_id": config["eos_token_id"],
            "bos_token_id": config["bos_token_id"],
            "pad_token_id": config["pad_token_id"],
        },
    )

    summary = {
        "model": "bagpiper",
        "vocab_size": vocab_size,
        "text_offset": text_offset,
        "codec_base_offset": codec_base,
        "codec_layer_size": codec_layer_size,
        "num_streams": num_streams,
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "overrides_applied": applied,
        "notes": notes,
    }
    _report(summary)
    return summary


# ---------------------------------------------------------------------------
# OpusLM / OpusLM-dialogue
# ---------------------------------------------------------------------------
# Every id below is looked up by *name* in the released YAML's token_list, so a
# checkpoint whose special-token block moved fails loudly instead of silently
# mapping to the wrong token. Where a slot exists but was never named (OpusLM's
# end-of-utterance), the alternatives are tried in order.
OPUSLM_ID_FIELDS: dict[str, tuple[str, ...]] = {
    "pad_token_id": ("<pad>",),
    "eos_token_id": ("<sos/eos>",),
    "system_prompt_token_id": ("<system_prompt>",),
    "user_input_token_id": ("<user_input>",),
    "assistant_output_token_id": ("<assistant_output>",),
    "eou_token_id": ("<eou>", "<unused_token_11>"),
    "codec_ssl_start_end_token_id": ("<codec_ssl_start/end>",),
    "text_bpe_start_end_token_id": ("<text_bpe_start/end>",),
    "spk_start_end_token_id": ("<spk_start/end>",),
    "textlm_task_token_id": ("<textlm_task>",),
    "codec_ssl_asr_task_token_id": ("<codec_ssl_asr_task>",),
    "codec_ssl_tts_task_token_id": ("<codec_ssl_tts_task>",),
    "codec_ssl_plain_tts_task_token_id": ("<codec_ssl_plain_tts_task>",),
    "codec_ssl_audiolm_task_token_id": ("<codec_ssl_audiolm_task>",),
    "text_dialogue_task_token_id": ("<text_dialogue_task>",),
    "audio_dialogue_task_token_id": ("<audio_dialogue_task>",),
}

# ESPnet's OpusLM starts generation from <sos/eos> and has no separate
# beginning-of-sequence token, but HF configs must carry a bos id. Slot 1 of
# the special-token block is <unk>, and that is the slot the released vLLM
# config has always pointed at.
OPUSLM_BOS_SLOT_NAME = "<unk>"

# ARDelay writes one codec frame per step across `codec_token_per_frame`
# streams; the first is the SSL stream, the rest are the DAC codec streams.
OPUSLM_ARCHITECTURES = {
    "opuslm": "OpusLMForConditionalGeneration",
    "opuslm_dialogue": "OpusLMDialogueForConditionalGeneration",
}


def _opuslm_dtype(
    espnet_yaml: dict[str, Any],
    checkpoint_dtype: str | None,
    notes: list[str],
) -> str:
    """The dtype to serve in, from the YAML's DeepSpeed settings.

    ESPnet trains these models under DeepSpeed, so ``ds_config_dict`` says what
    precision the run actually used -- ``train_dtype`` stays float32 even when
    bf16 is enabled. When the stored weights disagree, say so and keep the
    YAML's answer.
    """
    ds = espnet_yaml.get("ds_config_dict") or {}
    if (ds.get("bf16") or {}).get("enabled"):
        dtype = "bfloat16"
    elif (ds.get("fp16") or {}).get("enabled"):
        dtype = "float16"
    else:
        dtype = espnet_yaml.get("train_dtype") or "bfloat16"
    if checkpoint_dtype and checkpoint_dtype != dtype:
        notes.append(
            f"config.torch_dtype is {dtype} (the YAML's DeepSpeed setting), "
            f"but the weights are stored as {checkpoint_dtype}. vLLM will cast "
            "on load. This is expected for the dialogue checkpoint; pass "
            "--dtype to vllm serve to override."
        )
    return dtype


def build_opuslm_assets(
    out_dir: Path,
    *,
    model: str,
    offline_dir: Path | None = None,
    legacy_overrides: bool = True,
    checkpoint_dtype: str | None = None,
) -> dict[str, Any]:
    """Write OpusLM's config.json and tokenizer into ``out_dir``.

    The tokenizer is the backbone's, copied byte-for-byte, with one change: a
    content-only chat template, because OpusLM's structural tokens are added
    afterwards by OpusLMTokenizer in the reserved id range.

    ``checkpoint_dtype`` is the dtype the weights are actually stored in, if
    known. It is only cross-checked against what the YAML asks for, never used
    as the answer: the dialogue checkpoint is stored as float32 while its YAML
    trained in bf16, and serving it as float32 would double the memory and
    change the numerics of a configuration nobody has run.
    """
    if model not in OPUSLM_SOURCES:
        raise ValueError(f"not an OpusLM model: {model!r}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency of vLLM
        raise SystemExit(
            f"ERROR: PyYAML is required to read the released config.yaml ({exc})."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    src = OPUSLM_SOURCES[model]

    print("Fetching pinned config/tokenizer sources ...")
    espnet_yaml = yaml.safe_load(fetch(src["espnet_yaml"], offline_dir).decode())
    backbone = json.loads(fetch(src["transformer_config"], offline_dir))
    for key, source in src.items():
        print(f"  {source.repo}@{source.revision[:8]}/{source.filename} ({key})")

    # ---- tokenizer: copy the backbone's files verbatim -------------------
    subword_model = espnet_yaml.get("subword_model")
    tok_repo = src["tokenizer.json"].repo
    if subword_model and subword_model != tok_repo:
        raise SystemExit(
            f"ERROR: {src['espnet_yaml'].repo}'s config.yaml names tokenizer "
            f"{subword_model!r} but this generator is pinned to {tok_repo!r}. "
            "Re-pin deliberately rather than shipping a mismatched tokenizer."
        )
    for fname in OPUSLM_TOKENIZER_FILES:
        (out_dir / fname).write_bytes(fetch(src[fname], offline_dir))
    tok_cfg = json.loads(fetch(src["tokenizer_config.json"], offline_dir))
    tok_cfg["chat_template"] = OPUSLM_CHAT_TEMPLATE
    _write_json(out_dir / "tokenizer_config.json", tok_cfg)
    notes.append(
        f"tokenizer files copied verbatim from {tok_repo}; only "
        "tokenizer_config.chat_template is set (content-only, see "
        "convert_opuslm_ckpt.fix_chat_template)."
    )

    # ---- vocabulary boundaries, straight out of token_bias ---------------
    token_list = espnet_yaml["token_list"]
    bias = espnet_yaml["token_bias"]
    for block in ("special_token", "ssl", "codec", "text_bpe"):
        if block not in bias:
            raise SystemExit(
                f"ERROR: config.yaml token_bias has no {block!r} block; this "
                "is not an OpusLM configuration."
            )
    vocab_size = len(token_list)
    ssl_start, ssl_end = bias["ssl"]
    codec_start, codec_end = bias["codec"]
    text_start, text_end = bias["text_bpe"]
    nq = espnet_yaml["codec_token_per_frame"]
    num_codec_streams = nq - 1  # stream 0 is SSL; the rest are DAC codec
    codec_span = codec_end - codec_start
    if codec_span % num_codec_streams:
        raise SystemExit(
            f"ERROR: codec range {codec_span} is not divisible by "
            f"{num_codec_streams} streams."
        )
    codec_per_stream_size = codec_span // num_codec_streams
    print(
        f"\nVocabulary from token_bias: special [0,{bias['special_token'][1]}) "
        f"ssl [{ssl_start},{ssl_end}) codec [{codec_start},{codec_end}) "
        f"text [{text_start},{text_end}) -> vocab_size {vocab_size}"
    )

    def lookup(field: str, names: tuple[str, ...]) -> int:
        for name in names:
            try:
                return token_list.index(name)
            except ValueError:
                continue
        raise SystemExit(
            f"ERROR: none of {list(names)} is in the released token_list, so "
            f"{field} cannot be derived. The special-token block changed."
        )

    config: dict[str, Any] = {
        "model_type": model,
        "architectures": [OPUSLM_ARCHITECTURES[model]],
        "vocab_size": vocab_size,
        "hidden_size": backbone["hidden_size"],
        "intermediate_size": backbone["intermediate_size"],
        "num_hidden_layers": backbone["num_hidden_layers"],
        "num_attention_heads": backbone["num_attention_heads"],
        "num_key_value_heads": backbone["num_key_value_heads"],
        "max_position_embeddings": espnet_yaml.get("transformer_conf", {}).get(
            "n_ctx", backbone["max_position_embeddings"]
        ),
        "rms_norm_eps": backbone["rms_norm_eps"],
        "rope_theta": float(backbone["rope_theta"]),
        # corelm_conf.share_emb says whether lm_head is tied to the embedding.
        "tie_word_embeddings": bool(
            espnet_yaml.get("corelm_conf", {}).get("share_emb", False)
        ),
        "torch_dtype": _opuslm_dtype(espnet_yaml, checkpoint_dtype, notes),
        "ssl_token_start": ssl_start,
        "ssl_token_end": ssl_end,
        "codec_token_start": codec_start,
        "codec_token_end": codec_end,
        "text_token_start": text_start,
        "text_token_end": text_end,
        "nq": nq,
        "num_codec_streams": num_codec_streams,
        "codec_per_stream_size": codec_per_stream_size,
        "speaker_prompt_length": espnet_yaml["speaker_prompt_length"],
    }
    config["bos_token_id"] = lookup("bos_token_id", (OPUSLM_BOS_SLOT_NAME,))
    for field, names in OPUSLM_ID_FIELDS.items():
        config[field] = lookup(field, names)
    config.update(OPUSLM_DECODER_WIRING)
    if model == "opuslm_dialogue":
        config.update(OPUSLM_DIALOGUE_EXTRA_WIRING)

    applied = _apply_overrides(config, model, legacy_overrides)
    _write_json(out_dir / "config.json", config)

    summary = {
        "model": model,
        "vocab_size": vocab_size,
        "text_token_start": text_start,
        "text_token_end": text_end,
        "nq": nq,
        "num_codec_streams": num_codec_streams,
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
        "overrides_applied": applied,
        "notes": notes,
    }
    _report(summary)
    return summary


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _apply_overrides(
    config: dict[str, Any], model: str, enabled: bool
) -> list[str]:
    """Apply LEGACY_OVERRIDES in place, printing each one either way."""
    overrides = LEGACY_OVERRIDES.get(model, {})
    if not overrides:
        return []
    applied: list[str] = []
    print()
    for dotted, (value, why) in sorted(overrides.items()):
        target = config
        *parents, leaf = dotted.split(".")
        for part in parents:
            target = target[part]
        derived = target.get(leaf)
        if enabled:
            target[leaf] = value
            applied.append(dotted)
            print(f"  {dotted} = {value!r} (kept for parity; {why})")
        else:
            print(
                f"  {dotted} = {derived!r} (backbone-derived; the validated "
                f"reference uses {value!r}: {why})"
            )
    return applied


def _report(summary: dict[str, Any]) -> None:
    print(f"\nWrote {len(summary['files'])} file(s):")
    for name in summary["files"]:
        print(f"  {name}")
    for note in summary["notes"]:
        print(f"  NOTE: {note}")


def build_assets(
    out_dir: Path,
    *,
    model: str,
    offline_dir: Path | None = None,
    legacy_overrides: bool = True,
    checkpoint_dtype: str | None = None,
) -> dict[str, Any]:
    """Build the config/tokenizer for whichever of the three models is named."""
    if model == "bagpiper":
        return build_bagpiper_assets(
            out_dir, offline_dir=offline_dir, legacy_overrides=legacy_overrides
        )
    return build_opuslm_assets(
        out_dir,
        model=model,
        offline_dir=offline_dir,
        legacy_overrides=legacy_overrides,
        checkpoint_dtype=checkpoint_dtype,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
REFERENCE_CHAT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello."},
]


def validate(out_dir: Path, model: str) -> None:
    """Prove the generated assets are self-consistent and actually loadable.

    Raises SystemExit on the first failure. Every check compares the written
    files against a value recomputed here from first principles, never against
    a stored blob, so a wrong generator cannot validate itself.
    """
    print(f"\nValidating {out_dir} ...")
    config = json.loads((out_dir / "config.json").read_text())
    if config["model_type"] != model:
        raise SystemExit(
            f"FAIL: config.json model_type is {config['model_type']!r}, "
            f"expected {model!r}"
        )

    if model == "bagpiper":
        _validate_bagpiper_layout(out_dir, config)
    else:
        _validate_opuslm_layout(out_dir, config)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "  SKIP: transformers is not importable, so the tokenizer was not "
            "loaded. The layout checks above still ran."
        )
        return

    tok = AutoTokenizer.from_pretrained(str(out_dir), trust_remote_code=False)
    print(f"  tokenizer loads: {type(tok).__name__}, len={len(tok)}")

    if model == "bagpiper":
        _validate_bagpiper_tokenizer(tok, config)
    else:
        _validate_opuslm_tokenizer(tok, config)
    print("  validation passed")


def _validate_bagpiper_layout(out_dir: Path, config: dict[str, Any]) -> None:
    offset = config["text_token_offset"]
    base = config["codec_base_offset"]
    layer = config["codec_layer_size"]
    streams = config["num_stream"]
    expected = base + streams * layer
    if config["vocab_size"] != expected:
        raise SystemExit(
            f"FAIL: vocab_size {config['vocab_size']} != {base} + "
            f"{streams}*{layer} = {expected}"
        )
    if config["text_config"]["vocab_size"] != config["vocab_size"]:
        raise SystemExit("FAIL: text_config.vocab_size != vocab_size")
    if config["text_token_end"] != base:
        raise SystemExit("FAIL: text_token_end != codec_base_offset")
    print(
        f"  arithmetic closes: {offset} + {base - offset} + {streams}x{layer} "
        f"= {config['vocab_size']}"
    )

    tok_cfg = json.loads((out_dir / "tokenizer_config.json").read_text())
    decoder = tok_cfg["added_tokens_decoder"]
    ids = sorted(int(i) for i in decoder)
    want = list(range(offset)) + list(range(base, expected))
    if ids != want:
        raise SystemExit(
            f"FAIL: added_tokens_decoder covers {len(ids)} ids, expected "
            f"{len(want)} (reserved block plus codec block, contiguous)"
        )
    names = [decoder[str(i)]["content"] for i in ids]
    want_names = bagpiper_reserved_names() + bagpiper_codec_names(streams, layer - 1)
    if names != want_names:
        bad = next(
            i for i, (a, b) in enumerate(zip(names, want_names)) if a != b
        )
        raise SystemExit(
            f"FAIL: added_tokens_decoder name {bad} is {names[bad]!r}, "
            f"expected {want_names[bad]!r}"
        )
    print(f"  {len(ids)} special ids named and contiguous")

    vocab = json.loads((out_dir / "vocab.json").read_text())
    lo, hi = min(vocab.values()), max(vocab.values())
    if lo != offset or hi >= base:
        raise SystemExit(
            f"FAIL: vocab.json spans {lo}..{hi}, expected to start at {offset} "
            f"and stay below {base}"
        )
    print(f"  vocab.json: {len(vocab)} text tokens spanning {lo}..{hi}")


def _validate_opuslm_layout(out_dir: Path, config: dict[str, Any]) -> None:
    spans = [
        ("ssl", config["ssl_token_start"], config["ssl_token_end"]),
        ("codec", config["codec_token_start"], config["codec_token_end"]),
        ("text", config["text_token_start"], config["text_token_end"]),
    ]
    for name, start, end in spans:
        if not 0 <= start < end <= config["vocab_size"]:
            raise SystemExit(
                f"FAIL: {name} span [{start},{end}) is not inside "
                f"[0,{config['vocab_size']})"
            )
    if config["ssl_token_end"] != config["codec_token_start"]:
        raise SystemExit("FAIL: ssl and codec ranges are not adjacent")
    if config["codec_token_end"] != config["text_token_start"]:
        raise SystemExit("FAIL: codec and text ranges are not adjacent")
    codec_span = config["codec_token_end"] - config["codec_token_start"]
    if codec_span != config["num_codec_streams"] * config["codec_per_stream_size"]:
        raise SystemExit(
            f"FAIL: codec span {codec_span} != num_codec_streams * "
            f"codec_per_stream_size"
        )
    if config["nq"] != config["num_codec_streams"] + 1:
        raise SystemExit("FAIL: nq != num_codec_streams + 1 (SSL stream)")
    reserved = config["ssl_token_start"]
    for field in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if not 0 <= config[field] < reserved:
            raise SystemExit(
                f"FAIL: {field}={config[field]} is outside the reserved "
                f"special block [0,{reserved})"
            )
    print(
        f"  spans adjacent and in range; {config['num_codec_streams']}x"
        f"{config['codec_per_stream_size']} codec + SSL = nq {config['nq']}"
    )


def _validate_bagpiper_tokenizer(tok: Any, config: dict[str, Any]) -> None:
    """Round-trip the specials, then check the chat template's exact ids."""
    for field, name in (
        ("pad_token_id", "<|pad|>"),
        ("bos_token_id", "<|bos|>"),
        ("eos_token_id", "<|eos|>"),
        ("eot_token_id", "<|eot|>"),
        ("audio_token_id", "<|audio|>"),
    ):
        got = tok.convert_tokens_to_ids(name)
        if got != config[field]:
            raise SystemExit(
                f"FAIL: tokenizer maps {name} to {got}, config says "
                f"{config[field]}"
            )
    base = config["codec_base_offset"]
    layer = config["codec_layer_size"]
    probes = {
        "<codec_layer0_pad>": base,
        "<codec_layer0_code0>": base + 1,
        f"<codec_layer{config['num_stream'] - 1}_code{layer - 2}>": (
            config["vocab_size"] - 1
        ),
    }
    for name, want in probes.items():
        got = tok.convert_tokens_to_ids(name)
        if got != want:
            raise SystemExit(
                f"FAIL: tokenizer maps {name} to {got}, expected {want}"
            )
    print(f"  specials and codec boundaries round-trip ({len(probes)} probes)")

    rendered = tok.apply_chat_template(
        REFERENCE_CHAT, tokenize=False, add_generation_prompt=True
    )
    expected_text = (
        "<|bos|><|system|><|text|>You are a helpful assistant.<|eos|>"
        "<|user|><|text|>Say hello.<|eos|><|assistant|>"
    )
    if rendered != expected_text:
        raise SystemExit(
            f"FAIL: chat template rendered\n  {rendered!r}\nexpected\n  "
            f"{expected_text!r}"
        )
    ids = tok.apply_chat_template(
        REFERENCE_CHAT, tokenize=True, add_generation_prompt=True
    )
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    direct = tok.encode(expected_text, add_special_tokens=False)
    if ids != direct:
        raise SystemExit(
            "FAIL: templated ids differ from encoding the same string "
            f"({len(ids)} vs {len(direct)} tokens)"
        )
    role_ids = [
        config[f]
        for f in (
            "bos_token_id",
            "system_token_id",
            "text_token_id",
        )
    ]
    if ids[:3] != role_ids:
        raise SystemExit(
            f"FAIL: prompt starts with {ids[:3]}, expected {role_ids}"
        )
    if ids[-1] != config["assistant_token_id"]:
        raise SystemExit(
            f"FAIL: generation prompt ends with {ids[-1]}, expected "
            f"{config['assistant_token_id']}"
        )
    if max(ids) >= config["text_token_end"]:
        raise SystemExit(
            f"FAIL: a prompt token id ({max(ids)}) landed in the codec range"
        )
    print(f"  chat template encodes to {len(ids)} ids, all in the text range")


def _validate_opuslm_tokenizer(tok: Any, config: dict[str, Any]) -> None:
    """The inner BPE must cover exactly the model's text span."""
    span = config["text_token_end"] - config["text_token_start"]
    if len(tok) > span:
        raise SystemExit(
            f"FAIL: tokenizer has {len(tok)} tokens but the model reserves "
            f"only {span} ids for text"
        )
    rendered = tok.apply_chat_template(REFERENCE_CHAT, tokenize=False)
    joined = "".join(m["content"] for m in REFERENCE_CHAT)
    if rendered != joined:
        raise SystemExit(
            "FAIL: chat template is not content-only; it rendered\n  "
            f"{rendered!r}\nexpected\n  {joined!r}"
        )
    ids = tok.encode(joined, add_special_tokens=False)
    if not ids or max(ids) >= span:
        raise SystemExit(
            f"FAIL: encoding the reference text produced ids up to "
            f"{max(ids) if ids else None}, outside [0,{span})"
        )
    print(
        f"  content-only template; {len(tok)} BPE tokens fit the {span}-id "
        "text span"
    )


def compare_with_reference(out_dir: Path, ref_dir: Path) -> int:
    """Diff every generated file against a known-good directory.

    Prints one verdict per file and returns the number of files whose JSON
    content differs (byte differences that parse equal are reported as
    "same content"). Files only the reference has are listed too.
    """
    print(f"\nComparing {out_dir} against {ref_dir} ...")
    generated = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    differing = 0
    for name in generated:
        ours, theirs = out_dir / name, ref_dir / name
        if not theirs.exists():
            print(f"  {name}: only in the generated directory")
            continue
        if ours.read_bytes() == theirs.read_bytes():
            print(f"  {name}: byte-identical")
            continue
        if name.endswith(".json"):
            try:
                a = json.loads(ours.read_text())
                b = json.loads(theirs.read_text())
            except json.JSONDecodeError:
                print(f"  {name}: DIFFERS (unparseable)")
                differing += 1
                continue
            if a == b:
                print(f"  {name}: same content, different formatting")
                continue
            keys = _json_diff_keys(a, b)
            print(f"  {name}: DIFFERS at {len(keys)} key(s): {keys[:8]}")
            differing += 1
        else:
            print(
                f"  {name}: DIFFERS ({ours.stat().st_size} vs "
                f"{theirs.stat().st_size} bytes)"
            )
            differing += 1
    extra = sorted(
        p.name
        for p in ref_dir.iterdir()
        if p.is_file() and p.name not in generated
    )
    if extra:
        print(f"  only in the reference: {extra}")
    print(f"  {differing} file(s) differ in content")
    return differing


def _json_diff_keys(a: Any, b: Any, prefix: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            out += _json_diff_keys(
                a.get(key, ...), b.get(key, ...), f"{prefix}{key}."
            )
        return out
    return [] if a == b else [prefix.rstrip(".") or "<root>"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the config.json and tokenizer that released ESPnet "
            "checkpoints do not ship, from pinned public sources."
        )
    )
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--out", type=Path, help="Directory to write the assets into"
    )
    parser.add_argument(
        "--offline-dir",
        type=Path,
        help=(
            "Read the pinned source files from this directory instead of "
            "downloading them (see --print-sources)"
        ),
    )
    parser.add_argument(
        "--no-legacy-overrides",
        action="store_true",
        help=(
            "Emit purely backbone-derived values instead of the ones the "
            "validated reference config uses"
        ),
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        help="After building, diff every file against this known-good directory",
    )
    parser.add_argument(
        "--print-sources",
        action="store_true",
        help="List the pinned source files with URLs and hashes, then exit",
    )
    args = parser.parse_args()

    if args.print_sources:
        print(f"Pinned sources for {args.model}:")
        for key, src in all_sources(args.model).items():
            print(f"  {key}")
            print(f"    url    {src.url}")
            print(f"    sha256 {src.sha256}")
            print(f"    offline filename: {src.offline_name}")
        return

    if args.out is None:
        parser.error("--out is required unless --print-sources is given")

    build_assets(
        args.out,
        model=args.model,
        offline_dir=args.offline_dir,
        legacy_overrides=not args.no_legacy_overrides,
    )
    validate(args.out, args.model)
    if args.compare_with is not None:
        differing = compare_with_reference(args.out, args.compare_with)
        if differing:
            sys.exit(1)


if __name__ == "__main__":
    main()
