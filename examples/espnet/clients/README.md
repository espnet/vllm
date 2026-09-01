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

# TTS: text segment + audio segment (no stop_token_ids)
python client_bagpiper.py --task tts \
    --prompt "Read this aloud in a calm voice: The quick brown fox \
jumps over the lazy dog." --out tts.wav

# TTS with classifier-free guidance (server creates a shadow request)
python client_bagpiper.py --task tts_cfg --cfg 3.0 \
    --prompt "A dog barking twice in a quiet room." --out tts_cfg.wav
```

`tts` and `tts_cfg` default `--system` to `You are a helpful assistant.`
Keep it. Bagpiper picks its own output mode, and the system message is
what makes it render audio instead of answering with text alone —
measured 7/8 requests with it, 0/8 without. `--system ''` opts out.

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
    --text "How are you today?" --out reply.wav

# With a speaker prompt controlling the assistant voice
python client_opuslm_dialogue.py --task audio_dialogue \
    --speaker-audio /path/to/speaker.wav --audio /path/to/test.wav
```

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
