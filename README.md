# ⚡ flash-voice

**Talk with your own LLM – and interrupt it like a person.** A local voice stack you put in front of any
OpenAI-compatible model (Ollama, llama.cpp, vLLM, LM Studio, OpenAI, OpenRouter …).

[Deutsch](README.de.md)

- **Barge-in that works:** it listens the whole time, also while it speaks. Say something and it gets quieter at
  once; if it was only "mhm", a cough, its own echo or someone else in the room, it carries on – if you really want
  something, it stops and hears you **from your first syllable**.
- **It knows when you are done:** at the first short pause the decider asks "is the thought finished?".
  "What's the weather tomorrow in Vienna?" → the answer starts right away. "I'd like, uhm …" → it waits.
- **Small:** the whole voice stack runs in **~1.2 GB of GPU memory, or on the CPU alone** – the decider is
  [flash-turn](https://huggingface.co/apptor/flash-turn), a 73 MB model that answers in ~11 ms on one CPU core.
- **Voice chat for Open WebUI:** one `docker compose up` gives [Open WebUI](https://github.com/open-webui/open-webui)
  a voice and ears, CPU only ([below](#open-webui-voice-chat-in-one-command)).
- **Local and free to use:** every part is Apache 2.0, MIT or CC BY. Nothing is stored on the server; the conversation
  lives in your browser.

## How small can it get

flash-voice is built from parts. Each one is a small service of its own, so you pick what fits your hardware. The
LLM itself is not counted here: it can run anywhere (same machine, another one, a cloud API).

| Part | Choice | GPU memory | Notes |
|---|---|---|---|
| **Page + ear** (always) | flash-voice, Silero VAD | – | Python, CPU only |
| **Decider** | [flash-turn](https://huggingface.co/apptor/flash-turn) (default) | – | English + German, 73 MB, ~11 ms on one CPU core |
| | vllm-d3-mini 4B (optional) | ~5 GB | the big one |
| **Ears** (speech → text) | [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) q8_0 | **~1.2 GB** | English, German, Croatian; ~30 ms per sentence; fewest errors in our test |
| | Fluid Speech Pro int8 on the CPU | – | same model, ~1.4 s per sentence on 4 cores |
| | Whisper large-v3-turbo | ~1.2 GB | ~100 languages, ~0.4 s per sentence |
| **Voice** (text → speech) | **lite**: Phonon English + [apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts-at) German | – | CPU, ~0.5 s per sentence; German also with an Austrian accent |
| | Qwen3-TTS 1.7B | ~4.6 GB | 10 languages, best sound, voice cloning |

**Packages:**

| Package | Parts | GPU memory | Start |
|---|---|---|---|
| **Lite** | Fluid + lite voice + flash-turn | **~1.2 GB** – or **none** with the ears on the CPU | `TTS=lite STT=fluid` |
| **Full** | Qwen3-TTS + Fluid (or Whisper) + flash-turn | ~6 GB | `STT=fluid` |
| **Open WebUI** | Fluid (CPU) + lite voice in one container | none, ~1 GB RAM | `docker compose up -d` |

The page, the ear and the decider work the same in every package. What the Full package adds: the Qwen voice sounds
more natural, speaks more languages and can clone voices.

## How it works

```
 Browser / phone (one page)                         flash-voice (Python, CPU)
 ┌───────────────────────────┐   WebSocket 16 kHz   ┌────────────────────────────────────────────┐
 │ microphone (echo cancel.) │ ───────────────────► │ the ear: Silero VAD every 32 ms            │
 │ orb, transcript, text box │ ◄─── speech / end /  │   → the ears write it down                 │
 │ plays the answer, ducks   │      duck / barge    │   → the decider: finished? interruption?   │
 └───────────────────────────┘                      │ the turn: your LLM (stream) → sentences →  │
              ▲  NDJSON: text + audio pieces        │   the voice, piece by piece                │
              └──────────────────────────────────── └────────────────────────────────────────────┘
                                                        │ voice :8086       │ ears :8087       │ decider :8088
                                                        ▼                   ▼                  ▼
                                                    lite        CPU     Fluid    1.2 GB    flash-turn   CPU
                                                    Qwen3   4.6 GB      Whisper  1.2 GB    or vllm-d3-mini
                                                                                           ~5 GB, or rules
```

| Moment | What happens | Time |
|---|---|---|
| You stop talking | first pause (0.4 s) → ears + decider "finished?" | answer starts ~1.3–1.5 s after your last word, recognition included |
| "I'd like, uhm …" | decider: not finished → waits up to 1.8 s of silence | |
| You talk while it speaks | voice for 160 ms → it gets quieter (25 %) | ~0.2 s |
| … it was "mhm" / echo / noise | ears + rules + decider → full volume again | ~1 s |
| … it was an interruption | decider: interruption → it stops and listens on | ~1.5 s after you started |

The decider answers two questions: *is the thought finished?* and *what is the person doing while the assistant
speaks?* (interrupting / just listening / talking to someone else). By default it is
[flash-turn](https://huggingface.co/apptor/flash-turn), see [The decider](#the-decider). Without any decider, simple
rules take over: 1.1 s of silence ends your turn, and words that are not "mhm", "yes" … interrupt.

## Open WebUI: voice chat in one command

flash-voice lite is a small container that gives [Open WebUI](https://github.com/open-webui/open-webui) a voice and
ears – English and German, CPU only, no GPU, ~1 GB of RAM. It speaks the OpenAI audio API, which Open WebUI already
understands: no plugin to install, nothing to configure by hand.

```bash
curl -O https://raw.githubusercontent.com/ViktorsGitUser/flash-voice/main/docker/lite/docker-compose.yml
docker compose up -d          # then open http://localhost:3000 and press the headset button in a chat
```

The compose file starts Open WebUI already wired to it (and to Ollama on your computer – change `OLLAMA_BASE_URL` or
add an OpenAI-compatible API in Open WebUI's settings). **Already running Open WebUI?** Start only the voice:

```bash
docker build -t flash-voice-lite -f docker/lite/Dockerfile https://github.com/ViktorsGitUser/flash-voice.git#main
docker run -d --name flash-voice-lite --restart unless-stopped -p 8090:8090 -v flash-voice-models:/models flash-voice-lite
```

and in Open WebUI: *Admin Panel → Settings → Audio*: speech-to-text engine **OpenAI**, URL
`http://<host>:8090/v1`, key `none`, model `whisper-1`; text-to-speech engine **OpenAI**, same URL and key, model
`tts-1`, voice `m2` (or `f2`; `m2-at` for the Austrian accent). Then the microphone button transcribes, the speaker
button reads answers aloud and the headset button starts a hands-free call.

Measured on 4 CPU cores: a sentence is spoken after ~0.3 s (German) / ~0.5–0.9 s (English) and recognised in
~1.4 s; the container needs ~0.9 GB of RAM, the models ~1 GB of disk (downloaded on the first start). Open WebUI's
call mode has its own, simpler turn-taking; the interruptions and the decider described here are in flash-voice's own
page. The same API works for any app that speaks the OpenAI audio API (`services/openai_audio.py`).

## Plug it into your own setup

| You want … | What you do | Effort |
|---|---|---|
| **a voice for the LLM you already run** | install, set `FV_LLM_URL` in `.env` (Ollama, LM Studio, llama.cpp, OpenAI …), start, open the page | ~15 min |
| **voice in your own app or UI** | talk to flash-voice over `WS /api/ear` and `POST /api/turn` (below); `web/index.html` is a complete example | an afternoon |
| **only one part** (e.g. just text-to-speech or speech-to-text) | each service is plain HTTP and runs alone (below) | minutes |

**The ear and the turn** (flash-voice, port 8099):

- **`WS /api/ear`** – send the microphone as 16 kHz 16-bit mono PCM (binary frames, any size) and
  `{"t":"mode","mode":"listening|thinking|speaking|off"}` / `{"t":"said","text":…}` (what your assistant is saying,
  for the echo filter). You get `{"e":"ready"|"speech"|"reset"|"end"|"duck"|"resume"|"barge", …}`; `end` carries
  `text` and `lang`.
- **`POST /api/turn`** – `{"messages":[{"role","content"}…], "voice", "lang", "speak":true}` → NDJSON lines
  `{"t":"delta","text"}`, `{"t":"audio","text","audio":<base64 WAV>,"cont"}`, `{"t":"done","text"}`.
- Or use only the ear and bring your own LLM and TTS.

**The services alone:**

```bash
curl -F "text=Guten Morgen!" -F speaker=m2 http://127.0.0.1:8086/speak -o hello.wav   # voice: text -> WAV
curl -F audio=@question.webm -F langs=de,en http://127.0.0.1:8087/api/stt               # ears: -> {"text","lang"}
```

The voice services also have `/speak_stream` (WAV pieces as they are made) and `/voices`; the ears `/api/words` (when
each word is said, for subtitles). For the German voice without any server, in the browser or in a few lines of
Python, see [apptor-tiny-tts-at](https://huggingface.co/apptor/apptor-tiny-tts-at).

What it does not have (yet): a Docker image or a pip package. Each part has its own Python environment (so torch
is only needed for the Qwen voice); tested on Linux and Windows with NVIDIA GPUs.

## Install

```bash
git clone https://github.com/ViktorsGitUser/flash-voice && cd flash-voice
cp .env.example .env      # set FV_LLM_URL (and FV_LLM_MODEL / FV_LLM_KEY)

# the page and the ear (always; no GPU)
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# the ears (both choices use this environment)
python -m venv .venv-stt && .venv-stt/bin/pip install -r services/requirements-stt.txt
#   Fluid Speech Pro on the GPU: parakeet.cpp as a library + the model (~0.9 GB)
git clone https://github.com/mudler/parakeet.cpp && cmake -S parakeet.cpp -B parakeet.cpp/build -DPARAKEET_SHARED=ON -DGGML_CUDA=ON
cmake --build parakeet.cpp/build --target parakeet -j
.venv-stt/bin/pip install huggingface_hub && .venv-stt/bin/hf download FluidVoice-Labs/fluid-speech-pro-gguf fluid-speech-pro-q8_0.gguf --local-dir models
#   Whisper needs nothing more (downloaded on the first start, ~1.6 GB)

# the decider: flash-turn (73 MB, CPU)
.venv/bin/pip install huggingface_hub && .venv/bin/hf download apptor/flash-turn --local-dir models/flash-turn

# the voice - one of:
#   lite: Phonon English + apptor-tiny-tts German (CPU, no torch; the same environment as the Open WebUI container)
python -m venv .venv-lite && .venv-lite/bin/pip install -r docker/lite/requirements.txt
.venv-lite/bin/hf download apptor/apptor-tiny-tts-at apptor-tiny-tts-at.onnx apptor-tiny-tts-at.json apptor-g2p-de.onnx apptor-g2p-de.json apptor-vocos-de.int8.onnx --local-dir models
#   Qwen3-TTS (GPU, torch; the model is downloaded on the first start, ~4 GB)
python -m venv .venv-tts && .venv-tts/bin/pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-tts/bin/pip install -r services/requirements-tts.txt

# optional, instead of flash-turn: the big decider in llama.cpp's llama-server (b11719 or newer), ~5 GB GPU
.venv-stt/bin/hf download ggml-org/vllm-d3-mini-GGUF vllm-d3-mini-Q8_0.gguf --local-dir models
```

On Windows use `.venv-tts\Scripts\pip` etc.

## Start

```bash
TTS=lite STT=fluid scripts/start-all.sh                                 # Lite
STT=fluid scripts/start-all.sh                                          # Full (without STT=: Whisper)
FV_DECIDER=llm LLAMA_SERVER=/path/to/llama-server scripts/start-all.sh  # with the big decider
# Windows: scripts\start-all.bat with the same variables (set TTS=lite etc.)
```

`TTS` = `lite` | `qwen` (default) | `phonon`, `STT` = `fluid` | `whisper` (default), `FV_DECIDER` = `auto` (default:
flash-turn when `models/flash-turn` is there) | `mini` | `llm` | `rules`. The script uses the environments above when
they exist (`TTS_PY`, `STT_PY`, `FV_PY` to point elsewhere). Open the page, tap the microphone, talk. Tap the orb to
interrupt by hand. Settings (top right): voice, interrupt by voice on/off, theme, clear the conversation.

## The parts in detail

### Ears: Fluid Speech Pro (German, English, Croatian)

`services/fluid_stt_server.py`: [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) by
FluidVoice (CC BY 4.0, 0.6B parameters, built on NVIDIA Parakeet TDT 0.6B v3) through
[parakeet.cpp](https://github.com/mudler/parakeet.cpp) (MIT). On Common Voice German it made fewer mistakes than
Whisper large-v3-turbo (word error rate 6.4 % vs 7.9 % for speakers from Germany, 5.2 % vs 6.9 % with an Austrian
accent) and is ~10x faster. The model does not report the language; the server picks it from `langs` by the words it
recognised. Only these three languages, no `auto` – for other languages use Whisper.
`FV_FLUID_LIB` = path to `libparakeet.so` (the start script finds `parakeet.cpp/build/libparakeet.so`).
Without a GPU: `FV_FLUID_BACKEND=onnx` with the int8 ONNX export (FluidVoice-Labs/fluid-speech-pro-onnx,
`FV_FLUID_MODEL=<folder>`, `pip install onnx-asr onnxruntime`), ~0.9 s per sentence on four CPU threads.
Fluid Speech Pro is CC BY 4.0, not Apache: keep the attribution (see NOTICE) when you ship it.

### Ears: Whisper large-v3-turbo

`services/stt_server.py` (faster-whisper, MIT): `FV_STT_MODEL`, `FV_STT_COMPUTE`, `FV_STT_LANGS`. Any language
(`langs=auto,de,en`: Whisper picks, the hints only when it is unsure); `/api/words` takes the spoken text as a hint.

### Voice: apptor-tiny-tts (German, CPU, Apache 2.0)

`services/tinytts_server.py` with apptor-tiny-tts, a German model built on
[tiny-voca](https://github.com/n4ze3m/tiny-voca) by n4ze3m (not an official tiny-voca release; weights:
[apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts) and, with an Austrian accent switch,
[apptor-tiny-tts-at](https://huggingface.co/apptor/apptor-tiny-tts-at); browser demo:
[tiny-tts.apptor.at](https://tiny-tts.apptor.at)): 4.5M parameters, one 21 MB ONNX file, about 40x faster than real time
on four CPU threads, no GPU. Letters -> phonemes with the small apptor-g2p-de model, so **no eSpeak NG** is needed.
Two voices, `f2` (female) and `m2` (male); the Phonon voice ids are accepted too. Model and code are **Apache 2.0,
commercial use allowed**. Trained with the tiny-voca recipe on ~12 hours of synthetic German speech from Qwen3-TTS
(texts: Common Voice, CC0). German only: `FV_TINYTTS_EN_URL=http://127.0.0.1:8085` passes English sentences on to
Phonon's English model. Alone: `FV_TINYTTS_MODEL=models/apptor-tiny-tts-at.onnx .venv-tinytts/bin/python services/tinytts_server.py` (port 8089).

**Austrian accent** (apptor-tiny-tts-at, v0.2, still mild): the voice ids `f2-de`/`m2-de` speak Hochdeutsch, `-at` with
accent 1.0, `-wien` 1.3, `-oarg` 1.6; plain `f2`/`m2` use `FV_TINYTTS_ACCENT` (default 1.3). A form field `accent`
(0 to 1.6) overrides it per request. `models/apptor-tiny-tts-de.onnx` (from apptor/apptor-tiny-tts) is the plain German model.

The second stage, [Vocos](https://github.com/gemelo-ai/vocos) (MIT) fine-tuned on the two voices, re-synthesises the
output for a clearly cleaner sound. It is on by default when `models/apptor-vocos-de.int8.onnx` is there (ONNX, no
torch, ~0.1 s per sentence); `FV_TINYTTS_VOCOS=` (empty) switches it off. The older PyTorch version
(`=models/apptor-tiny-tts-vocos-de.pt` or `=pretrained`) needs torch and `services/requirements-tinytts-vocos.txt`.
`FV_TINYTTS_G2P=espeak` uses the espeak-ng program instead of the G2P model.

### Voice: Phonon (CPU, German + English)

`services/phonon_server.py` with [Phonon](https://huggingface.co/gradium/phonon-31m-en) by Gradium: about 7x faster
than real time on an 8-core CPU, the first sound after ~0.2 s. Voices are German/English pairs (Annika/Harper,
Lotta/Zoey, Nele/Everly, Henrik/Preston, Leon/Trevor, Mika/Easton), so a voice stays itself when the language
changes. No voice cloning. Alone: `HF_TOKEN=hf_... .venv-phonon/bin/python services/phonon_server.py` (port 8085).

**Licenses differ:** `phonon-31m-en` is Apache 2.0. The German model `phonon-40m-de` is **CC BY-NC 4.0 – non-
commercial use only** – and gated on Hugging Face (accept its terms, then set `HF_TOKEN` or `FV_HF_TOKEN_FILE`).
For commercial use in German take apptor-tiny-tts or the Qwen voice.

### Voice: Qwen3-TTS 1.7B (GPU)

`services/tts_server.py`: `FV_TTS_MODEL` (default `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`: 9 built-in voices);
`FV_TTS_PRIORITY` (default `above`, also `high` / `normal`) – the voice must stay ahead of real time while the LLM
writes on the same PC: on an 8-core CPU it took 1.2x real time with normal priority (stutter) and 0.5x with "above
normal". With the `…-1.7B-Base` model and `FV_VOICES=<folder>` it runs a **voice library**: every recording in the
folder is a voice (see [voices/README.md](voices/README.md)), and the library is managed over HTTP –
`POST /voices/clone` (a recording becomes a voice, hiss and hum filtered), `/voices/mix` (a weighted mixture of
voices, saved as a speaker embedding), `/voices/update`, `/voices/delete`; `/speak` also takes `instruct` (a speaking
style), `mix` (preview a mixture) and `temperature`. Labels: a voice's `name.json`, or `names.json` in the folder.

### The decider

**flash-turn** (default, [apptor/flash-turn](https://huggingface.co/apptor/flash-turn), Apache 2.0): one small text
model for both questions, English and German. Built on distilbert-base-multilingual-cased (vocabulary cut to English +
German), fine-tuned on ~2.8 million German and ~2 million English sentences (whole = finished, cut after a word = not
finished) and ~21,000 interjections while an assistant speaks (written by a local LLM and checked by vllm-d3-mini, plus
people addressed by name). ONNX int8, 73 MB, ~11 ms per question on one CPU thread – no GPU.

Measured on test sets never used in training; vllm-d3-mini asked the way flash-voice asks it (Q8_0, llama-server):

| Test | flash-turn | vllm-d3-mini |
|---|---|---|
| End of turn, decider test cases (English / German, 24 each) | 23 / 23 | 24 / 23 |
| Interjections, decider test cases (English / German, 28 each) | **26 / 25** | 24 / 18 |
| End of turn, 300 clips of Austrian speakers (speech recognition text) | **92.3 %** | 87.0 % |
| End of turn, unseen sentences (English / German) | 91.5 % / 91.0 % | 90.7 % / 90.7 % |
| Time per question | **11 ms on 1 CPU thread** | ~35 ms on a GPU |
| Size | **73 MB** | ~5 GB |

The ear asks it at every first pause and every interjection. Longer interjections (4+ words) still stop the assistant
unless flash-turn is very sure (≥ 80 %) that they are meant for someone else – it rather stops once too often than
talks over you. A wrong "not finished" only costs time: the ear then waits up to 1.8 s of silence instead of 1.1 s.

**vllm-d3-mini** (optional, `FV_DECIDER=llm`): a 4B decision model in llama.cpp's `llama-server` (b11719 or newer: it
has the `/v1/systemone` endpoint and knows d3-mini's decision format), ~5 GB of GPU memory, ~35 ms.
`scripts/start-decider.sh` runs it (`LLAMA_SERVER`, `DECIDER_GGUF`); flash-voice asks it at `FV_DECIDER_URL`. The old
names `FV_CLEF_URL`, `NO_CLEF` and `CLEF_GGUF` still work.

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
| `FV_DECIDER` | `auto` | `mini` (flash-turn) / `llm` (vllm-d3-mini) / `rules`; auto = mini if `models/flash-turn` is there |
| `FV_TTS_URL` / `FV_STT_URL` / `FV_DECIDER_URL` | `:8086` / `:8087` / `:8088` | the big decider's URL (`FV_DECIDER=llm`) |
| `FV_LANGS` | `en,de` | what the ears choose from; the decider's questions follow the language heard |
| `FV_VOICE` | `serena` | a voice id of the voice service (unknown ids: its default voice) |
| `FV_HOST` / `FV_PORT` | `127.0.0.1` / `8099` | |
| `FV_PASSWORD` | – | set it when other devices can reach the page |
| `FV_CERT` / `FV_KEY` | – | paths, or `FV_CERT=self` |

## Known limits

- While a GPU voice is still generating, the ears share the GPU with it: an interruption is then confirmed in
  ~1.5–3 s instead of ~1 s (the volume goes down right away regardless). With a CPU voice this does not happen.
- Whether a sentence is meant for someone else in the room is often impossible to tell from the words alone
  ("could you pass the water?"). Clear cases (a name or "honey", the TV, a sports commentator) are recognised; when in
  doubt it rather stops once too often than talks over you.
- flash-turn knows English and German; in other languages the pause rules decide.
- The page keeps the conversation in the browser (last 60 messages). There are no tools, files or memory – this is the
  voice, bring your own brain.

## Licenses

flash-voice is Apache 2.0 (see [LICENSE](LICENSE)). The models and libraries it uses keep their own licenses – all
permissive except the optional German Phonon model (CC BY-NC 4.0, non-commercial; the German apptor-tiny-tts voice is
the Apache 2.0 alternative); Fluid Speech Pro is CC BY 4.0 (attribution), see [NOTICE](NOTICE). Only clone voices of
people who agreed to it.
