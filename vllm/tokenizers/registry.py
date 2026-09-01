# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
from typing_extensions import TypeVar, assert_never

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.transformers_utils.config import _maybe_register_hf_config, get_config
from vllm.transformers_utils.repo_utils import (
    any_pattern_in_repo_files,
    is_mistral_model_repo,
)
from vllm.utils.import_utils import resolve_obj_by_qualname

from .hf import CachedHfTokenizer
from .protocol import TokenizerLike

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig, RunnerType

logger = init_logger(__name__)


# Model types whose hub tokenizer_class is incorrect and should be overridden with
# TokenizersBackend (the generic fast tokenizer). Adding a model type here is always a
# temporary workaround and better long term solutions are:
# - Add model type to MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS in transformers (better)
# - Fix tokenizer_class on the hub for the affected models (best)
_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS: set[str] = {
    "internlm2",
    "step3_vl",
    "step3p7",
    "unlimited-ocr",
}

_VLLM_TOKENIZERS = {
    # ``cohere`` mode uses the standard cached HF tokenizer; only the
    # renderer (template stage) is replaced with a melody-based one.
    "cohere": ("hf", "CachedHfTokenizer"),
    "deepseek_v32": ("deepseek_v32", "DeepseekV32Tokenizer"),
    "deepseek_v4": ("deepseek_v4", "DeepseekV4Tokenizer"),
    "hf": ("hf", "CachedHfTokenizer"),
    "kimi_audio": ("kimi_audio", "KimiAudioTokenizer"),
    "kimi_k3": ("hf", "CachedHfTokenizer"),
    "mistral": ("mistral", "MistralTokenizer"),
    "opuslm": ("opuslm", "OpusLMTokenizer"),
    "opuslm_dialogue": ("opuslm_dialogue", "OpusLMDialogueTokenizer"),
    # Inkling uses the plain HF tokenizer for token operations; the "inkling"
    # mode exists to select the InklingRenderer, which renders chat to
    # token ids natively (Inkling has no Jinja chat template).
    "inkling": ("hf", "CachedHfTokenizer"),
}


@dataclass
class _TokenizerRegistry:
    # Tokenizer mode ->  (tokenizer module, tokenizer class)
    tokenizers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def register(self, tokenizer_mode: str, module: str, class_name: str) -> None:
        if tokenizer_mode in self.tokenizers:
            logger.warning(
                "%s.%s is already registered for tokenizer_mode=%r. "
                "It is overwritten by the new one.",
                module,
                class_name,
                tokenizer_mode,
            )

        self.tokenizers[tokenizer_mode] = (module, class_name)

        return None

    def load_tokenizer_cls(self, tokenizer_mode: str) -> type[TokenizerLike]:
        if tokenizer_mode not in self.tokenizers:
            raise ValueError(f"No tokenizer registered for {tokenizer_mode=!r}.")

        module, class_name = self.tokenizers[tokenizer_mode]
        logger.debug_once(f"Loading {class_name} for {tokenizer_mode=!r}")

        return resolve_obj_by_qualname(f"{module}.{class_name}")

    def load_tokenizer(self, tokenizer_mode: str, *args, **kwargs) -> TokenizerLike:
        tokenizer_cls = self.load_tokenizer_cls(tokenizer_mode)
        return tokenizer_cls.from_pretrained(*args, **kwargs)


TokenizerRegistry = _TokenizerRegistry(
    {
        mode: (f"vllm.tokenizers.{mod_relname}", cls_name)
        for mode, (mod_relname, cls_name) in _VLLM_TOKENIZERS.items()
    }
)


def resolve_tokenizer_args(
    tokenizer_name: str | Path,
    *args,
    runner_type: "RunnerType" = "generate",
    tokenizer_mode: str = "auto",
    model_type: str | None = None,
    **kwargs,
):
    revision: str | None = kwargs.get("revision")
    download_dir: str | None = kwargs.get("download_dir")

    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        from modelscope.hub.snapshot_download import snapshot_download

        # avoid circular import
        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not Path(tokenizer_name).exists():
            # Use file lock to prevent multiple processes from
            # downloading the same file at the same time.
            with get_lock(tokenizer_name, download_dir):
                tokenizer_path = snapshot_download(
                    model_id=str(tokenizer_name),
                    cache_dir=download_dir,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    # Ignore weights - we only need the tokenizer.
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
                )
                tokenizer_name = tokenizer_path

    if "truncation_side" not in kwargs:
        if runner_type == "generate" or runner_type == "draft":
            kwargs["truncation_side"] = "left"
        elif runner_type == "pooling":
            kwargs["truncation_side"] = "right"
        else:
            assert_never(runner_type)

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")

        tokenizer_mode = "hf"
        kwargs["use_fast"] = False

    # OpusLM uses a shifted global vocabulary layout:
    # tokenizer IDs are mapped to [text_token_start, text_token_end) at runtime.
    if tokenizer_mode == "auto" and model_type == "opuslm":
        tokenizer_mode = "opuslm"
    if tokenizer_mode == "auto" and model_type == "opuslm_dialogue":
        tokenizer_mode = "opuslm_dialogue"

    # Try to use official Mistral tokenizer if possible
    if (
        tokenizer_mode == "auto"
        and is_mistral_model_repo(
            model_name_or_path=str(tokenizer_name), revision=revision
        )
        and any_pattern_in_repo_files(
            model_name_or_path=str(tokenizer_name),
            allow_patterns=["tekken.json", "tokenizer.model.v*"],
            revision=revision,
        )
    ):
        tokenizer_mode = "mistral"

    # Fallback to HF tokenizer
    if tokenizer_mode == "auto":
        tokenizer_mode = "hf"

    return tokenizer_mode, tokenizer_name, args, kwargs


cached_resolve_tokenizer_args = lru_cache(resolve_tokenizer_args)


# The kwargs the two OpusLM tokenizers pop in ``from_pretrained``, as
# (kwarg suffix, hf_config field, fallback). Injecting a name a tokenizer does
# not pop would forward it to transformers and raise there, so these tables have
# to stay in step with the two ``from_pretrained`` bodies. The task tokens and
# stream count are here because the tokenizer lays ESPnet's task layout over
# text-only prompts; prompts that carry audio get the same layout from
# OpusLMMultiModalProcessor, which reads the config directly. Both sides go
# through OpusLMTaskLayout, so they agree.
_OPUSLM_COMMON_TOKENIZER_KWARGS = (
    ("text_token_offset", "text_token_start", 13448),
    ("pad_token_id", "pad_token_id", 0),
    ("eos_token_id", "eos_token_id", 5),
    ("codec_ssl_start_end_token_id", "codec_ssl_start_end_token_id", 34),
    ("text_bpe_start_end_token_id", "text_bpe_start_end_token_id", 35),
    ("textlm_task_token_id", "textlm_task_token_id", 64),
    ("codec_ssl_asr_task_token_id", "codec_ssl_asr_task_token_id", 80),
    ("codec_ssl_tts_task_token_id", "codec_ssl_tts_task_token_id", 81),
    ("codec_ssl_plain_tts_task_token_id", "codec_ssl_plain_tts_task_token_id", 82),
    ("codec_ssl_audiolm_task_token_id", "codec_ssl_audiolm_task_token_id", 83),
    ("text_dialogue_task_token_id", "text_dialogue_task_token_id", 88),
    ("audio_dialogue_task_token_id", "audio_dialogue_task_token_id", 89),
    ("nq", "nq", 9),
)

_OPUSLM_TOKENIZER_KWARGS = {
    # The two released checkpoints have different text vocabulary sizes, so
    # text_token_end differs in its fallback. Both configs carry the field, so
    # the fallback only decides what a config that omits it gets.
    "opuslm": _OPUSLM_COMMON_TOKENIZER_KWARGS
    + (("text_token_end", "text_token_end", 113800),),
    # The dialogue tokenizer additionally needs the three role markers, because
    # it is the one that builds multi-turn dialogue layouts.
    "opuslm_dialogue": _OPUSLM_COMMON_TOKENIZER_KWARGS
    + (
        ("text_token_end", "text_token_end", 62600),
        ("system_prompt_token_id", "system_prompt_token_id", 8),
        ("user_input_token_id", "user_input_token_id", 9),
        ("assistant_output_token_id", "assistant_output_token_id", 10),
    ),
}


def _inject_opuslm_tokenizer_kwargs(
    model_type: str, hf_config, kwargs: dict
) -> dict:
    """Fill OpusLM tokenizer kwargs from the model's hf_config.

    Both OpusLM tokenizers configure their global-vocab shift and their ESPnet
    task layout from ``<model_type>_``-prefixed kwargs, falling back to the
    released checkpoints' values when a kwarg is absent. Reading them off the
    config means a checkpoint that renumbers a task token still gets the layout
    its own config describes, instead of silently getting the released one. A
    kwarg the caller passed explicitly still wins.
    """
    kwargs = dict(kwargs)
    for suffix, field, default in _OPUSLM_TOKENIZER_KWARGS[model_type]:
        value = getattr(hf_config, field, None)
        kwargs.setdefault(
            f"{model_type}_{suffix}",
            default if value is None else int(value),
        )
    return kwargs


def tokenizer_args_from_config(config: "ModelConfig", **kwargs):
    model_type = getattr(config.hf_config, "model_type", None)
    if model_type in _OPUSLM_TOKENIZER_KWARGS:
        kwargs = _inject_opuslm_tokenizer_kwargs(model_type, config.hf_config, kwargs)

    return cached_resolve_tokenizer_args(
        config.tokenizer,
        runner_type=config.runner_type,
        tokenizer_mode=config.tokenizer_mode,
        model_type=model_type,
        revision=config.tokenizer_revision,
        trust_remote_code=config.trust_remote_code,
        **kwargs,
    )


_T = TypeVar("_T", bound=TokenizerLike, default=TokenizerLike)


def get_tokenizer(
    tokenizer_name: str | Path,
    *args,
    tokenizer_cls: type[_T] = TokenizerLike,  # type: ignore[assignment]
    trust_remote_code: bool = False,
    revision: str | None = None,
    download_dir: str | None = None,
    **kwargs,
) -> _T:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope."""
    if envs.VLLM_USE_FASTOKENS:
        # Process-global, idempotent patch that swaps the Rust BPE backend
        # of any HF fast tokenizer loaded afterwards. No-op for non-HF modes.
        from .fastokens import apply_fastokens_patch

        apply_fastokens_patch()

    tokenizer_mode, tokenizer_name, args, kwargs = cached_resolve_tokenizer_args(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        revision=revision,
        download_dir=download_dir,
        **kwargs,
    )

    if tokenizer_cls == TokenizerLike:
        tokenizer_cls_ = TokenizerRegistry.load_tokenizer_cls(tokenizer_mode)
    else:
        tokenizer_cls_ = tokenizer_cls

    # Ensure that, if the config were to come from vllm.transformers_utils.config, it is
    # registered with AutoConfig before the tokenizer is loaded. This is necessary since
    # tokenizer_cls_.from_pretrained will call AutoConfig.from_pretrained internally.
    # This may fail for paths that don't have a model config (e.g. LoRA adapters),
    # which is fine — those don't need custom config registration.
    # HF-backed tokenizers must receive the HF config. In a dual-format Mistral
    # repository, auto detection intentionally prefers params.json, but passing
    # that generic config to AutoTokenizer can select the wrong tokenizer class.
    config_format = "hf" if tokenizer_cls_ is CachedHfTokenizer else "auto"
    config = None
    with contextlib.suppress(ValueError, OSError):
        config = get_config(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            config_format=config_format,
        )

    # Some models have an incorrect tokenizer_class on the hub.
    # For these model types, bypass AutoTokenizer and use TokenizersBackend directly.
    model_type = getattr(config, "model_type", None) if config else None
    if model_type in _MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        logger.debug(
            "Overriding tokenizer_class to TokenizersBackend for model_type=%r",
            model_type,
        )
        tokenizer_cls_ = TokenizersBackend

    if config is not None and tokenizer_cls_ is CachedHfTokenizer:
        # AutoTokenizer otherwise reloads config.json internally. Reuse the
        # config that get_config just loaded successfully so a concurrent Hub
        # cache refresh cannot invalidate the file between the two reads.
        kwargs.setdefault("config", config)

    tokenizer = tokenizer_cls_.from_pretrained(tokenizer_name, *args, **kwargs)
    if model_type in _MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS:
        from vllm.tokenizers.hf import get_cached_tokenizer

        tokenizer = get_cached_tokenizer(tokenizer)
    if not tokenizer.is_fast:
        logger.warning(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    return tokenizer  # type: ignore


cached_get_tokenizer = lru_cache(get_tokenizer)


def cached_tokenizer_from_config(model_config: "ModelConfig", **kwargs):
    if model_config.skip_tokenizer_init:
        return None

    _maybe_register_hf_config(getattr(model_config, "hf_config", None))

    # OpusLM: this function is the path that actually constructs the
    # tokenizer used by the engine and the renderer, so the model_type-based
    # tokenizer_mode resolution and the config-driven kwargs must be applied
    # here as well (resolve_tokenizer_args only sees what we pass down).
    hf_config = getattr(model_config, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    if model_type in _OPUSLM_TOKENIZER_KWARGS:
        kwargs = _inject_opuslm_tokenizer_kwargs(model_type, hf_config, kwargs)
        kwargs.setdefault("model_type", model_type)

    return cached_get_tokenizer(
        model_config.tokenizer,
        runner_type=model_config.runner_type,
        tokenizer_mode=model_config.tokenizer_mode,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )
