# Reference clients

All clients target the OpenAI-compatible `/v1/chat/completions` endpoint
of a server started with the sibling `serve_*.sh` scripts (default ports:
bagpiper 9811, opuslm 9812, opuslm_dialogue 9813). Audio-producing tasks
save the returned WAV to `--out` (default `out.wav`) and print its
duration and sample rate. Replace `/path/to/test.wav` with any 16 kHz
mono WAV (e.g. a short speech clip).

## client_bagpiper.py

```bash
# Plain text generation (stops at <|eot|>)
python client_bagpiper.py --task text --prompt "What is 2+2?"

# Audio understanding (input_audio content part)
python client_bagpiper.py --task audio_understand \
    --audio /path/to/test.wav --prompt "What sound is in this audio?"

# Audio generation: text segment + audio segment (no stop_token_ids).
# The built-in default prompt is already an in-distribution example.
python client_bagpiper.py --task tts --out tts.wav

# To make it say a specific line, DESCRIBE THE SCENE and quote the line.
python client_bagpiper.py --task tts --out tts.wav \
    --prompt "A calm male voice, close-miked in a quiet studio, says: \
'Your package will arrive on Tuesday.' No background noise."

# Same, with classifier-free guidance (server creates a shadow request)
python client_bagpiper.py --task tts_cfg --cfg 3.0 --out tts_cfg.wav
```

Bagpiper generates audio from a natural-language scene description, with
spoken content quoted inside the description. Use this format to specify
the voice, recording environment, and speech content.

`tts` and `tts_cfg` use the audio-generation system prompt
`DEFAULT_TTS_SYSTEM` by default. `--system ''` disables it. Example prompts
and generated audio are available in [demo_assets](../demo_assets/README.md).

## client_opuslm.py

```bash
# Plain TTS (task token 82)
python client_opuslm.py --task tts --prompt "Hello world" --out tts.wav

# Voice-cloning TTS (task token 81): add a voice prompt
python client_opuslm.py --task tts --prompt "Hello world" \
    --audio /path/to/test.wav --out tts_clone.wav

# ASR (task token 80)
python client_opuslm.py --task asr --audio /path/to/test.wav

# Text-LM continuation (task token 64)
python client_opuslm.py --task textlm --prompt "Once upon a time"
```

## client_opuslm_dialogue.py

```bash
# Spoken dialogue: user audio in, assistant audio out (task token 89)
python client_opuslm_dialogue.py --task audio_dialogue \
    --audio /path/to/test.wav --out reply.wav

# Text dialogue (task token 88)
python client_opuslm_dialogue.py --task text_dialogue \
    --text "How are you today?"

# With a speaker prompt controlling the assistant voice
python client_opuslm_dialogue.py --task audio_dialogue \
    --speaker-audio /path/to/speaker.wav --audio /path/to/test.wav
```

Text dialogue returns text without audio. Use `audio_dialogue` for a spoken
response.

## client_bagpiper_stress.py

Concurrent stress test: fires text-only, audiogen, and audiogen+CFG
workloads at the same time (128 connections per task type by default).
Input is ESPnet triplet JSONL:
`{"example_id": ..., "messages": [[role, "text"|"audio", content], ...]}`
where audio content is a local file path.

```bash
python client_bagpiper_stress.py \
    --audiogen-input /path/to/audiogen.jsonl \
    --text-input /path/to/text.jsonl \
    --output-dir output_stress --save-random 10
```

Results are written to `stress_{audiogen_cfg,audiogen,text}.jsonl` plus
random WAV samples under `output_stress/audio/`.
