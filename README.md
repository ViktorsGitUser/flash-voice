# ⚡ flash-voice

**Talk with your own LLM – and interrupt it like a person.** A local voice stack you put in front of any
OpenAI-compatible model (Ollama, llama.cpp, vLLM, LM Studio, OpenAI, OpenRouter …).

[Deutsch](README.de.md)

- **Barge-in that works:** it listens the whole time, also while it speaks. Say something and it gets quieter at
  once; if it was only "mhm", a cough, its own echo or someone else in the room, it carries on – if you really want
  something, it stops and hears you **from your first syllable**.
- **It knows when you are done:** at the first short pause a small decision model asks "is the thought finished?".
  "What's the weather tomorrow in Vienna?" → the answer starts right away. "I'd like, uhm …" → it waits.
- **Fast:** the answer is spoken sentence by sentence while the LLM is still writing; the first sound comes ~1 s
  after the first sentence is written. A "Hi!" plays the moment you start, with no delay.
- **Local and free to use:** every part is Apache 2.0 or MIT. Nothing is stored on the server; the conversation lives
  in your browser.

## How it works

```
 Browser / phone (one page)                         flash-voice (Python, no GPU)
 ┌───────────────────────────┐   WebSocket 16 kHz   ┌────────────────────────────────────────────────┐
 │ microphone (echo cancel.) │ ───────────────────► │ the ear: Silero VAD every 32 ms                │
 │ orb, transcript, text box │ ◄─── speech / end /  │   → Whisper writes down → Clef decides         │
 │ plays the answer, ducks   │      duck / barge    │     (finished? interruption? just "mhm"?)       │
 └───────────────────────────┘                      │ the turn: your LLM (stream) → sentences →      │
              ▲  NDJSON: text + audio pieces        │   Qwen3-TTS, piece by piece                    │
              └──────────────────────────────────── └────────────────────────────────────────────────┘
                                                        │ TTS :8086     │ STT :8087     │ Clef :8088
                                                        ▼               ▼               ▼
                                                   Qwen3-TTS 1.7B   Whisper large-   clef-flash
                                                   (~5 GB VRAM)     v3-turbo (~1.5)  (~6 GB, optional)
```

| Moment | What happens | Time |
|---|---|---|
| You stop talking | first pause (0.4 s) → Whisper + Clef "finished?" | answer starts ~1.3–1.5 s after your last word, recognition included |
| "I'd like, uhm …" | Clef: not finished → waits up to 1.8 s of silence | |
| You talk while it speaks | voice for 160 ms → it gets quieter (25 %) | ~0.2 s |
| … it was "mhm" / echo / noise | Whisper + rules + Clef → full volume again | ~1 s |
| … it was an interruption | Clef: interruption → it stops and listens on | ~1.5 s after you started |

Clef ([Cloudflare clef-flash](https://huggingface.co/Cloudflare/clef-flash)) is not a chat model: it gets a text and
questions with fixed answers and returns a probability for each, in ~0.3 s. Two questions are asked here – *is the
thought finished?* (yes/no) and *what is the person doing while the assistant speaks?* (interrupting / just
listening / talking to someone else). In tests: "mhm" → 96 % listening, "wait, stop" → 96 % interrupting.
Without Clef, simple rules take over (it works, just less cleverly).

## What you need

- An **NVIDIA GPU with ~13 GB free** for voice + ears + Clef (tested on an RTX 5060 Ti 16 GB, next to nothing else).
  Without Clef ~7 GB. The LLM itself can run anywhere – on the same machine, another one, or a cloud API.
- Python 3.12, an OpenAI-compatible LLM endpoint, and for Clef [llama.cpp](https://github.com/ggml-org/llama.cpp)
  b11556 or newer (its `llama-server` has the `/v1/systemone` endpoint).
- A browser – and **https** for the microphone (see [The microphone needs https](#the-microphone-needs-https)).

## Install

```bash
git clone https://github.com/ViktorsGitUser/flash-voice && cd flash-voice

# 1) the voice (its own environment: torch)
python -m venv .venv-tts && .venv-tts/bin/pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-tts/bin/pip install -r services/requirements-tts.txt

# 2) the ears
python -m venv .venv-stt && .venv-stt/bin/pip install -r services/requirements-stt.txt

# 3) the page and the ear (no GPU)
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 3b) optional: Phonon, a fast CPU voice for German + English (no GPU, ~300 MB RAM per language)
python -m venv .venv-phonon && .venv-phonon/bin/pip install -r services/requirements-phonon.txt

# 4) Clef (optional): llama.cpp's llama-server + the model (~6 GB)
pip install -U huggingface_hub && hf download bartowski/Cloudflare_clef-flash-GGUF Cloudflare_clef-flash-Q4_K_M.gguf --local-dir models

cp .env.example .env      # set FV_LLM_URL (and FV_LLM_MODEL / FV_LLM_KEY)
```

On Windows use `.venv-tts\Scripts\pip` etc. The voice and Whisper models are downloaded on the first start (~4 GB
and ~1.6 GB, Hugging Face cache).

## Start

```bash
TTS_PY=.venv-tts/bin/python STT_PY=.venv-stt/bin/python FV_PY=.venv/bin/python LLAMA_SERVER=/path/to/llama-server scripts/start-all.sh
# Windows: scripts\start-all.bat (set TTS_PY, STT_PY, FV_PY, LLAMA_SERVER first if they are elsewhere)
# without Clef: NO_CLEF=1
```

Open the page, tap the microphone, talk. Tap the orb to interrupt by hand.
Settings (top right): voice (9 built-in Qwen voices), interrupt by voice on/off, theme, clear the conversation.

### Optional: Phonon instead of the Qwen voice (CPU, German + English)

`services/phonon_server.py` speaks the same API as the Qwen voice (`/speak`, `/speak_stream`, `/voices`) with
[Phonon](https://huggingface.co/gradium/phonon-31m-en) by Gradium: about 7x faster than real time on an 8-core CPU,
the first sound after ~0.2 s, no GPU memory at all (the Qwen voice takes ~4.6 GB). Voices are German/English pairs
(Annika/Harper, Lotta/Zoey, Nele/Everly, Henrik/Preston, Leon/Trevor, Mika/Easton), so a voice stays itself when
the language changes. No voice cloning: the open Phonon models have only their catalog voices.

```bash
HF_TOKEN=hf_... .venv-phonon/bin/python services/phonon_server.py      # port 8085 (FV_PHONON_PORT)
FV_TTS_URL=http://127.0.0.1:8085 ...                                    # point flash-voice at it
```

**Licenses differ:** `phonon-31m-en` is Apache 2.0. The German model `phonon-40m-de` is **CC BY-NC 4.0 – non-
commercial use only** – and gated on Hugging Face (accept its terms, then set `HF_TOKEN` or `FV_HF_TOKEN_FILE`).
For commercial use keep the Qwen voice for German.

## The microphone needs https

> ⚠️ Browsers give a web page the microphone **only over https** – or on `http://localhost` / `http://127.0.0.1`
> on the same computer. Opened any other way (`http://192.168.x.x:8099`, a hostname, another device, an app's
> built-in preview browser) the microphone stays off and the page tells you so.

| Where you open it | What to do |
|---|---|
| In a normal browser (Chrome, Edge, Firefox, Safari) **on the same computer** | `http://127.0.0.1:8099/` – nothing else needed |
| On your phone / another device, **anywhere** (easiest, a real certificate) | [Tailscale](https://tailscale.com): `tailscale serve --bg --https=8443 http://127.0.0.1:8099`, then `https://<machine>.<tailnet>.ts.net:8443/` – works on the road too |
| On your phone / another device **in the same Wi-Fi** | `.env`: `FV_HOST=0.0.0.0`, `FV_CERT=self`, `FV_PASSWORD=…`; open `https://<pc-ip>:8099/` and accept the certificate warning once |
| You already have a domain | a reverse proxy with a certificate (Caddy, nginx …) in front of `127.0.0.1:8099` |

In the car over Bluetooth the car does the echo cancellation – that is where it is the most fun.

## Settings (`.env`)

| Variable | Default | |
|---|---|---|
| `FV_LLM_URL` | `http://127.0.0.1:11434/v1` | any OpenAI-compatible endpoint |
| `FV_LLM_MODEL` / `FV_LLM_KEY` | – | empty model = the first the endpoint lists |
| `FV_LLM_EXTRA` | – | JSON added to each request, e.g. `{"temperature":0.7}`; thinking models: switch thinking off for speed (e.g. Qwen3 `{"chat_template_kwargs":{"enable_thinking":false}}`) |
| `FV_SYSTEM_PROMPT` | built-in | `{date}` = today |
| `FV_TTS_URL` / `FV_STT_URL` / `FV_CLEF_URL` | `:8086` / `:8087` / `:8088` | Clef empty = rules only |
| `FV_LANGS` | `en,de` | what Whisper chooses from; Clef's questions follow the language heard |
| `FV_VOICE` | `serena` | |
| `FV_HOST` / `FV_PORT` | `127.0.0.1` / `8099` | |
| `FV_PASSWORD` | – | set it when other devices can reach the page |
| `FV_CERT` / `FV_KEY` | – | paths, or `FV_CERT=self` |

TTS service: `FV_TTS_MODEL` (default `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`: 9 built-in voices); `FV_TTS_PRIORITY`
(default `above`, also `high` / `normal`) – the voice must stay ahead of real time while the LLM writes on the same
PC: on an 8-core CPU it took 1.2x real time with normal priority (stutter) and 0.5x with "above normal". With the
`…-1.7B-Base` model and `FV_VOICES=<folder>` it runs a **voice library**: every recording in the folder is a voice
(see [voices/README.md](voices/README.md)), and the library is managed over HTTP – `POST /voices/clone` (a recording
becomes a voice, hiss and hum filtered), `/voices/mix` (a weighted mixture of voices, saved as a speaker embedding),
`/voices/update`, `/voices/delete`; `/speak` also takes `instruct` (a speaking style), `mix` (preview a mixture) and
`temperature`. Labels: a voice's `name.json`, or `names.json` in the folder.
STT service: `FV_STT_MODEL`, `FV_STT_COMPUTE`, `FV_STT_LANGS`; besides `/api/stt` it has `/api/words` – when each word
is said, for word-by-word subtitles.

## Use it from your own app

The two parts are plain HTTP/WebSocket – you can put them behind your own UI:

- **`WS /api/ear`** – send the microphone as 16 kHz 16-bit mono PCM (binary frames, any size) and
  `{"t":"mode","mode":"listening|thinking|speaking|off"}` / `{"t":"said","text":…}` (what your assistant is saying,
  for the echo filter). You get `{"e":"ready"|"speech"|"reset"|"end"|"duck"|"resume"|"barge", …}`; `end` carries
  `text` and `lang`.
- **`POST /api/turn`** – `{"messages":[{"role","content"}…], "voice", "lang", "speak":true}` → NDJSON lines
  `{"t":"delta","text"}`, `{"t":"audio","text","audio":<base64 WAV>,"cont"}`, `{"t":"done","text"}`.
- Or use only the ear and bring your own LLM and TTS.

`web/index.html` is a complete, dependency-free example (~400 lines).

## Known limits

- While the voice is still generating, Whisper shares the GPU with it: an interruption is then confirmed in ~1.5–3 s
  instead of ~1 s (the volume goes down right away regardless).
- Clef tends to treat a sentence to someone else in the room ("could you pass the water?") as an interruption – it
  rather stops once too often than talks over you.
- The page keeps the conversation in the browser (last 60 messages). There are no tools, files or memory – this is the
  voice, bring your own brain.

## Licenses

flash-voice is Apache 2.0 (see [LICENSE](LICENSE)). The models and libraries it uses keep their own licenses – all
permissive except the optional German Phonon model (CC BY-NC 4.0, non-commercial), see [NOTICE](NOTICE). Only
clone voices of people who agreed to it.
