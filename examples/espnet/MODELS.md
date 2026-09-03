# Where the models come from, and what each needs

Audited against the live Hugging Face trees on 2026-09-03. Everything below is
either proven by a hash/tensor comparison or labelled as inference. Nothing here
is a guess.

## Short answer

| this fork's name | official source | published format | needs conversion? |
| --- | --- | --- | --- |
| `bagpiper` | [`espnet/bagpiper-tts-sft`](https://huggingface.co/espnet/bagpiper-tts-sft) (speech) / [`espnet/bagpiper-sft`](https://huggingface.co/espnet/bagpiper-sft) (general) / [`espnet/bagpiper`](https://huggingface.co/espnet/bagpiper) (base) | native ESPnet `model.pt`, `{"module": state_dict}` | **yes** |
| `opuslm` | [`espnet/OpusLM_7B_Anneal`](https://huggingface.co/espnet/OpusLM_7B_Anneal) | native ESPnet `model.pth` | **yes** |
| `opuslm_dialogue` | [`espnet/multi_turn_SDS_RLAIF`](https://huggingface.co/espnet/multi_turn_SDS_RLAIF) | native ESPnet `2epoch.pth` | **yes** |

None of them is directly vLLM-loadable. Every one ships raw ESPnet weights with
no `config.json`, no tokenizer and no safetensors. `espnet/bagpiper-sft`'s own
model card says it plainly: *"It is not a Transformers from_pretrained directory
and no vLLM compatibility is claimed."*

## bagpiper

Two SFT checkpoints exist and they behave differently. This matters more than it
looks:

- **`espnet/bagpiper-tts-sft`** — the natural-language-guided speech synthesis
  model, `pipeline_tag: text-to-speech`. Use this one for speech. Verified: 6 of
  6 scene-description requests returned audio, all `finish_reason=stop`. The demo
  clips in `demo_assets/` come from it.
- **`espnet/bagpiper-sft`** — the paper-selected *general* SFT checkpoint.
  Verified on the same requests through the same server: **0 of 6 returned
  audio.** It emitted reasoning prose with no `<think>` wrapper and stopped at
  `eos` without ever emitting `eot`, so the text→audio phase transition never
  fired. Its release ships `inference_text.yaml` and `inference_audio.yaml`, and
  the audio one sets `enforce_modality: ["text", "audio"]` — the ESPnet decoder
  *forces* the modality sequence. Our phase machine does not force it, so this
  checkpoint needs either that enforcement or a different prompt regime. Treat
  audio output from `bagpiper-sft` as unsupported here until that is resolved.
- **`espnet/bagpiper`** — the pre-trained base (`base.pt`), backbone
  `Qwen/Qwen3-8B-Base`. Not fine-tuned; not used for the demos.

Both SFT repos carry 1,382 tensors: 1,381 bf16 parameters plus one fp32
`vocab_weight` of shape `(vocab_size,)`. `vocab_weight` is a loss-weighting
vector the strict ESPnet loader requires; it is not a model parameter, and the
converter drops it explicitly rather than silently.

`espnet/bagpiper-sft`'s `MANIFEST.json` records where its weights came from:
repo `JinchuanTian/bagpiper_sft` (dataset), revision
`b11d5a0c11ad488edd04e3734d4bdff764977f57`, path
`exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_270000/global_step269985/mp_rank_00_model_states.pt`
— a DeepSpeed shard, which is why the converter accepts that filename too.

### The vocabulary is fully determined by the released YAML

`train_stage3_qwen3_base.yaml` gives `tokenizer_name: Qwen/Qwen3-8B-Base`
(151,936 ids) and 8 Xcodec streams. That fixes the whole layout, and it closes
exactly:

```
256 reserved specials
+ 151,936 text ids      -> text range [256, 152192)
+ 8 x 1025 codec tokens -> codec range [152192, 160392)
= 160,392                = config.json vocab_size
```

The known-good tokenizer's `added_tokens_decoder` holds 8,456 entries = 256 + 8,200,
which matches. This is what `bootstrap_assets.py` reconstructs, so no
pre-existing directory is needed — see "Building the missing files" below.

## opuslm

**Proven.** The `model.pth` we converted has sha256
`a8e1a0166265fee20ea830d11c996d90da7060723eda6c89bf1ffd25a4378e61`, byte-identical
to `espnet/OpusLM_7B_Anneal/model.pth` (14,886,005,245 bytes, same LFS oid).

Backbone `allenai/OLMo-2-1124-7B` per its `config.yaml`; token list 113,870
entries, matching our `vocab_size`. The sibling
`espnet/OpusLM_1.7B_Anneal` is the 1.7B OpusLM (backbone
`HuggingFaceTB/SmolLM2-1.7B`, token list 62,670) — still OpusLM, **not** the
dialogue model.

## opuslm_dialogue

**Proven: [`espnet/multi_turn_SDS_RLAIF`](https://huggingface.co/espnet/multi_turn_SDS_RLAIF)**,
revision `a784cde04ffb1e7e8e83044dab24b54d1c298429`, file `2epoch.pth`
(7,469,766,446 bytes, LFS oid
`f602b54b5680b57cb5dceceb2d7184630f086d7d85b58e7cfbbf882f509bc332`).

How it was proven, not guessed: the released `2epoch.pth` was downloaded, the
converter's key mapping applied, and every tensor compared against the
`opuslm_dialogue` safetensors this fork was developed against. Result — 221
source tensors, one `criterion.*` dropped, **220 of 220 matching bit-for-bit,
max |Δ| = 0.0**, identical shapes and dtypes (F32 on both sides), zero keys on
either side without a partner.

Its `config.yaml` fits: backbone `HuggingFaceTB/SmolLM2-1.7B-Instruct`, token
list 62,670 (our `vocab_size`), `model: dpo`, and
`output_dir: exp/speechlm_audio_dialogue_fisher_train_delay_..._dpo_dialogue_combined_full`
— a multi-turn spoken-dialogue model trained with DPO/RLAIF.

Attribution, stated carefully: this is consistent with the model having been
published by Siddhant Arora — HF user [`Siddhant`](https://huggingface.co/Siddhant)
(full name "Arora") is an `espnet` org member and multi-turn SDS with RLAIF is his
research area. The *repository identity above is proven*; who uploaded the file is
**not** something the HF API exposes per-file, so treat the person as
well-supported inference rather than proof.

Two dead ends worth recording so nobody repeats them: `espnet/OpusLM_1.7B_Anneal`
shares the SmolLM2-1.7B geometry and the exact 62,670 vocabulary, which makes it
look like a candidate — it is not; its weights are bf16/3.77 GB and provably a
different checkpoint. `espnet/speechlm_unified_v1_1.7B` is also a different
checkpoint (3,768,476,459 bytes). Architecture and vocabulary overlap is not
provenance.

## Building the missing files

espnet publishes weights only, so a vLLM directory additionally needs a
`config.json` and a tokenizer. `convert/bootstrap_assets.py` builds both from
public artifacts, which is what the converters call when you do not pass
`--ref-dir`. Every fetch is pinned to a git revision **and** to a recorded
sha256, so an upstream edit stops the conversion instead of quietly changing
your model.

| model | tokenizer built from | geometry from | vocabulary layout from |
| --- | --- | --- | --- |
| `bagpiper` | [`Qwen/Qwen3-8B-Base`](https://huggingface.co/Qwen/Qwen3-8B-Base) `@49e3418f`, ids shifted +256, then 8×1025 codec tokens appended | the same repo's `config.json`, plus `Qwen/Qwen3-Omni-30B-A3B-Instruct` `@26291f79` for the audio tower | the arithmetic above, from `train_bagpiper_tts.yaml` |
| `opuslm` | [`allenai/OLMo-2-1124-7B`](https://huggingface.co/allenai/OLMo-2-1124-7B) `@7df9a825`, **byte-for-byte** | the same repo's `config.json` | `config.yaml`'s `token_bias`; every special-token id looked up **by name** in its `token_list` |
| `opuslm_dialogue` | [`HuggingFaceTB/SmolLM-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B) `@d7449ff7`, **byte-for-byte** | `HuggingFaceTB/SmolLM2-1.7B-Instruct` `@31b70e2e` | same, from `multi_turn_SDS_RLAIF/config.yaml` |

Two details worth knowing. The OpusLM YAMLs name the tokenizer in
`subword_model` and the transformer in `transformer_conf.hf_model_tag`, and for
the dialogue model **those are different repositories** — SmolLM-1.7B provides
the tokenizer, SmolLM2-1.7B-Instruct the geometry. And the tokenizer is copied
verbatim with exactly one change: a content-only chat template, because
OpusLM's structural tokens are added afterwards by `OpusLMTokenizer` in the
reserved id range (see `convert_opuslm_ckpt.fix_chat_template`).

Only three things are not fetched, because they are properties of this fork
rather than of a checkpoint: the chat template, the names of Bagpiper's 256
reserved and 8,200 codec tokens, and which codec repo decodes the audio. Run
`bootstrap_assets.py --model <name> --print-sources` for the full pinned list
with URLs and hashes.

### Verified against the known-good directories

Generated assets were diffed against the directories this fork was developed
and tested with. `merges.txt`, `chat_template.jinja` and
`generation_config.json` come out byte-identical; `tokenizer.json`, `vocab.json`
and `tokenizer_config.json` parse equal (only JSON formatting differs); the
OpusLM tokenizers are byte-identical to their backbones. For Bagpiper the
loaded tokenizers agree on `bos`/`eos`/`pad` ids, on `len(tok)`, and on the
exact token ids for the 364-character audio-generation system prompt, an
audio-placeholder conversation and raw CJK/emoji text. For both OpusLM models
`get_vocab()` is identical and sampled id→token round-trips match.

Three files are deliberately **not** reproduced as they stand:

- `config.json` gets the canonical `model_type`/`architectures` names. The
  reference directories still carried `speechlm_audio_encoder` and
  `speechlm_text` in the nested configs; the generated values match what
  `configs/bagpiper.py` actually declares.
- `added_tokens.json` records Qwen's 26 specials at their real (shifted) ids
  151899–151924. The reference listed them unshifted. Harmless either way —
  transformers reads `tokenizer_config.added_tokens_decoder` for a fast
  tokenizer — but there is no reason to copy the error.
- `special_tokens_map.json` names `<|bos|>`/`<|eos|>`/`<|pad|>`. The reference
  carried a stray Qwen-chat map whose `eos_token` was `<|endoftext|>`, which is
  not this model's eos. Verified inert: both directories load with
  `eos_token_id == 2`.

### Values that disagree with the backbone

A few fields in the hand-written reference configs do not match the public
backbone. The generator keeps the reference value so a fresh conversion is
numerically identical to what was tested, and prints the backbone's value
beside it every run. `--no-legacy-overrides` emits the backbone-derived values
instead.

| model | field | kept | backbone says |
| --- | --- | --- | --- |
| `bagpiper` | `text_config.max_position_embeddings` | 40960 | 32768 (40960 is the Qwen3-8B release's value) |
| `bagpiper` | `audio_config.n_window` | 100 | 50 |
| `bagpiper` | `audio_config.n_window_infer` | 400 | 800 |
| `opuslm` | `rms_norm_eps` | 1e-05 | 1e-06 |

The OpusLM one is the only genuinely unexplained entry: OLMo-2's own
`config.json` says `1e-06`, and the hand-written OpusLM config has always said
`1e-05`. OpusLM was not re-run for this change, so the tested value stands.

Two derivations worth spelling out, because the obvious source is the wrong one:

- **`torch_dtype` comes from the YAML's `ds_config_dict.bf16.enabled`, not from
  the weights.** Both models train under DeepSpeed with bf16 enabled while
  `train_dtype` stays `float32`, so the DeepSpeed block is what says the
  precision. It matters for `opuslm_dialogue`: its `2epoch.pth` is *stored* as
  float32, so reading the dtype off the tensors would emit `float32`, double the
  serving memory and change the numerics of a configuration nobody has run. The
  generator keeps `bfloat16` and prints a note when the stored weights disagree.
- **OpusLM's five `*_task_token_id` fields are absent from the hand-written
  config.** The generator writes them, looked up by name in `token_list`
  (64/80/81/82/83) — the same values `OpusLMConfig` was already defaulting to,
  so behaviour is unchanged and the ids are now derived rather than assumed.
