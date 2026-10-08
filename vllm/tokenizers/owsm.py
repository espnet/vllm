# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SentencePiece tokenizer preserving ESPnet's explicit token-ID ordering."""

import json
from pathlib import Path

import regex as re
import sentencepiece as spm
from transformers import BatchEncoding

from vllm.transformers_utils.repo_utils import hf_api


class OWSMTokenizer:
    def __init__(self, path, *, truncation_side="left"):
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        self.tokens = (path / "tokens.txt").read_text().splitlines()
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        if len(self.token_to_id) != len(self.tokens):
            raise ValueError("OWSM token_list contains duplicate entries")
        self.sp = spm.SentencePieceProcessor(model_file=str(path / "bpe.model"))
        self.bos_token_id = config["bos_token_id"]
        self.eos_token_id = config["eos_token_id"]
        self.pad_token_id = config["pad_token_id"]
        self.unk_token_id = self.token_to_id.get("<unk>", self.sp.unk_id())
        self.truncation_side = truncation_side
        self.name_or_path = str(path)
        self.is_fast = False
        self.vocab_size = len(self.tokens)
        self.max_token_id = self.vocab_size - 1
        self.max_chars_per_token = max(map(len, self.tokens))
        self.all_special_tokens = [
            token
            for token in self.tokens
            if token.startswith("<") and token.endswith(">")
        ]
        self.all_special_ids = [self.token_to_id[t] for t in self.all_special_tokens]
        self._special_set = set(self.all_special_tokens)
        self._split = re.compile(
            "("
            + "|".join(
                re.escape(t)
                for t in sorted(self.all_special_tokens, key=len, reverse=True)
            )
            + ")"
        )

    @classmethod
    def from_pretrained(cls, path_or_repo_id, *args, **kwargs):
        path = Path(path_or_repo_id)
        if not path.is_dir():
            path = Path(
                hf_api().snapshot_download(
                    str(path_or_repo_id),
                    revision=kwargs.get("revision"),
                    cache_dir=kwargs.get("download_dir"),
                    allow_patterns=["config.json", "tokens.txt", "bpe.model"],
                )
            )
        return cls(path, truncation_side=kwargs.get("truncation_side", "left"))

    def __len__(self):
        return self.vocab_size

    def num_special_tokens_to_add(self):
        return 0

    def get_vocab(self):
        return self.token_to_id.copy()

    def get_added_vocab(self):
        return {t: self.token_to_id[t] for t in self.all_special_tokens}

    def encode(self, text, truncation=None, max_length=None, add_special_tokens=True):
        # Prefix symbols are already explicit in OWSM's inference protocol.
        ids = []
        for chunk in self._split.split(text):
            if chunk in self._special_set:
                ids.append(self.token_to_id[chunk])
            elif chunk:
                ids.extend(
                    self.token_to_id.get(t, self.unk_token_id)
                    for t in self.sp.encode(chunk, out_type=str)
                )
        if truncation and max_length is not None:
            ids = (
                ids[-max_length:]
                if self.truncation_side == "left"
                else ids[:max_length]
            )
        return ids

    def __call__(self, text, text_pair=None, **kwargs):
        if text_pair is not None:
            raise ValueError("OWSM does not accept paired text inputs")
        rows = [text] if isinstance(text, str) else text
        ids = [
            self.encode(
                row,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in {"truncation", "max_length", "add_special_tokens"}
                },
            )
            for row in rows
        ]
        return BatchEncoding({"input_ids": ids[0] if isinstance(text, str) else ids})

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.token_to_id.get(tokens, self.unk_token_id)
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self.tokens[ids]
        return [
            self.tokens[i]
            for i in ids
            if not skip_special_tokens or i not in self.all_special_ids
        ]

    def convert_tokens_to_string(self, tokens):
        chunks: list[str] = []
        pieces: list[str] = []
        for token in tokens:
            if token in self._special_set:
                if pieces:
                    chunks.append(self.sp.decode(pieces))
                    pieces = []
                chunks.append(token)
            else:
                pieces.append(token)
        if pieces:
            chunks.append(self.sp.decode(pieces))
        return "".join(chunks)

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        return self.convert_tokens_to_string(
            self.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)
        )

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("OWSM uses explicit language/task decoder prefixes")
