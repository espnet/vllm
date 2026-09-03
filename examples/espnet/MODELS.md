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
(151,936 tokens) and 8 Xcodec streams. That fixes the whole layout, and it
closes exactly:

```
256 reserved specials
+ 151,936 text tokens   -> text range [256, 152192)
+ 8 x 1025 codec tokens -> codec range [152192, 160392)
= 160,392                = config.json vocab_size
```

The known-good tokenizer's `added_tokens_decoder` holds 8,456 entries = 256 + 8,200,
which matches. So config and tokenizer are *derivable* from public artifacts in
principle. They are not published though, so today the converter copies them from
`--ref-dir`. See "Known gap" below.

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

## Known gap: config.json and tokenizer are not published

espnet publishes weights only. To build a vLLM directory you additionally need a
Bagpiper/OpusLM `config.json` plus the tokenizer files, and no espnet repo ships
them. The converters therefore take `--ref-dir`, a directory that already has
them.

This is a real gap, not a papered-over one. Closing it properly means generating
`config.json` from the released YAML (the arithmetic above shows it is fully
determined) and rebuilding the tokenizer from the public base model plus the
8,456 added tokens. That generator is not written yet.
