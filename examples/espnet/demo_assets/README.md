# Bagpiper demo audio

Real output from this fork, generated on one H100 80GB on 2026-09-03. Every
prompt below was hand-written for this demo — none comes from a training set or
any third-party corpus, so these files are safe to redistribute.

## How they were made

| | |
| --- | --- |
| model | [`espnet/bagpiper-tts-sft`](https://huggingface.co/espnet/bagpiper-tts-sft) rev `675e2fafccc7dd7205fad6f8fdc4451f9ee6f768` |
| conversion | `examples/espnet/convert/convert_bagpiper_ckpt.py` — 1381 tensors written bit-for-bit, `vocab_weight` dropped |
| reproducible? | yes, from the official checkpoint alone. Re-running the converter without `--ref-dir` produced safetensors shards with **identical sha256** to the ones that made these clips. Sampling is stochastic, so you will get different waveforms from the same weights, not these files. |
| runtime | this fork at vLLM 0.28.0, `serve_bagpiper.sh`, TP=1 |
| request | `--task tts`, `mode: text_audio`, `text_temperature 0.6`, `audio_temperature 0.8`, `audio_topk 20`, `max_tokens 12000` |
| system prompt | the client's `DEFAULT_TTS_SYSTEM` (the one the model was trained with) |
| format | 16 kHz, mono, PCM_16 |

All six requests returned audio with `finish_reason=stop`; none was truncated.

## The clips

Bagpiper takes a **scene description** with any spoken line quoted inside it —
not a bare sentence to read. These prompts are the shape it expects.

### `bagpiper_tts_hello_greeting.wav` — 1.40 s

> A clear, friendly female voice, close-miked in a quiet room, says: `'Hello, how are you today?'`. She speaks at a relaxed, natural pace with a warm tone and no background noise or music.

### `bagpiper_tts_hello_greeting_cfg3.wav` — 1.24 s

Same prompt, with `--task tts_cfg --cfg 3.0`. Included as a controlled CFG
on/off pair with the clip above. CFG 3 is what the released
`inference_audio.yaml` uses for the audio stream.

### `bagpiper_tts_train_announcement.wav` — 2.78 s

> A calm male station announcer speaks over a quiet platform: `'The next train to Boston departs from platform nine.'` His voice is clear and measured, lightly reverberant as if amplified in a large hall, with no music.

### `bagpiper_tts_weather_report.wav` — 4.16 s

> A neutral female broadcaster in a dry studio reads: `'Tomorrow will be cloudy with a high of eighteen degrees and light rain in the evening.'` She speaks steadily and clearly, close-miked, no background sound.

### `bagpiper_tts_numbers_and_date.wav` — 8.14 s

> A precise male voice, studio-recorded and close-miked, dictates slowly and clearly: `'Your order total is one thousand two hundred thirty-four dollars and fifty-six cents, shipping on July fourth.'` No background noise or music.

## Did it say the right words?

Transcribed with `openai/whisper-base` and compared against the quoted text.
This is an **auxiliary** check, not a human listening test — Whisper is small and
makes its own mistakes.

| clip | Whisper heard | word error |
| --- | --- | --- |
| hello_greeting | `Hello, how are you today?` | **0.0%** |
| hello_greeting_cfg3 | `Hello, how are you today?` | **0.0%** |
| weather_report | `Tomorrow will be cloudy with a high of 18 degrees and light rain in the evening.` | 6.2% |
| train_announcement | `The next train to Boston departs from Platform 9.` | 11.1% |
| numbers_and_date | `Your order total is $1,234.56, shipping on July 4.` | 63.2% |

Read those numbers carefully. Every non-zero figure is **numeral orthography**,
not a speech error: Whisper writes "18" where the prompt said "eighteen",
"Platform 9" for "platform nine", and `$1,234.56` for the spoken amount. The
last row scores badly by raw word overlap while being semantically exact — the
model did say the money and the date correctly. After normalising digits to
words the content matches throughout.

## Measured properties

Checked, not assumed: all five files are 16 kHz mono PCM_16 with the WAV byte
identity `size == 44 + 2*frames`; none is silent (peak 0.53–1.00, active frames
72–96%); implied speaking rates are 2.1–4.0 words per second, inside the natural
band for read-aloud English.

Three clips touch full scale on a handful of samples (2, 2 and 7 samples
respectively) — audible clipping is unlikely at that count, but it is there and
worth knowing rather than hiding.

## What was not done

Nobody sat and listened to these files as part of producing them. The claims
above are measurements plus the Whisper cross-check. Judge the quality by ear.
