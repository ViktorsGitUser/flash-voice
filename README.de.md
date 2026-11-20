# ⚡ flash-voice

**Mit dem eigenen LLM reden – und ihm ins Wort fallen wie einem Menschen.** Ein lokaler Sprach-Stack vor jedem
OpenAI-kompatiblen Modell (Ollama, llama.cpp, vLLM, LM Studio, OpenAI, OpenRouter …).

[English](README.md)

- **Unterbrechen, das funktioniert:** Es hört die ganze Zeit zu, auch während es spricht. Sagst du etwas, wird es
  sofort leiser; war es nur „mhm“, ein Husten, das eigene Echo oder jemand anderer im Raum, redet es weiter – willst du
  wirklich etwas, hört es auf und hört dich **ab deiner ersten Silbe**.
- **Es merkt, wann du fertig bist:** Bei der ersten kurzen Pause fragt ein kleines Entscheidungsmodell „ist der
  Gedanke fertig?“. „Wie wird das Wetter morgen in Wien?“ → die Antwort beginnt sofort. „Ich hätte gern, ähm …“ → es wartet.
- **Schnell:** Die Antwort wird Satz für Satz gesprochen, während das LLM noch schreibt. Ein „Hi!“ kommt ohne
  Verzögerung, sobald du startest.
- **Lokal und frei verwendbar:** Alle Teile sind Apache 2.0 oder MIT. Der Server speichert nichts; das Gespräch liegt
  in deinem Browser.

## So funktioniert es

| Moment | Was passiert | Zeit |
|---|---|---|
| Du hörst auf zu reden | erste Pause (0,4 s) → Whisper + Clef „fertig?“ | Antwort beginnt ~1,3–1,5 s nach deinem letzten Wort, Spracherkennung inklusive |
| „Ich hätte gern, ähm …“ | Clef: nicht fertig → wartet bis 1,8 s Stille | |
| Du redest, während es spricht | 160 ms Stimme → es wird leiser (25 %) | ~0,2 s |
| … war „mhm“ / Echo / Geräusch | Whisper + Regeln + Clef → wieder laut | ~1 s |
| … war eine Unterbrechung | Clef: Unterbrechung → es hört auf und hört weiter zu | ~1,5 s nach deinem ersten Wort |

Clef, das Entscheidungsmodell ([vllm-d3-mini](https://huggingface.co/vllm-sr/d3-mini), Apache 2.0, 4B; bis November
2026 [Cloudflare clef-flash](https://huggingface.co/Cloudflare/clef-flash), 9B), ist kein Chat-Modell: Es bekommt einen
Text und Fragen mit festen Antworten und liefert für jede eine Wahrscheinlichkeit, in ~35 ms auf der GPU. Hier werden zwei Fragen
gestellt – *ist der Gedanke fertig?* und *was tut der Mensch, während der Assistent spricht?* (unterbricht / hört nur
zu / redet mit jemand anderem). Im Test: „mhm“ → 96 % zuhören, „warte mal, stopp“ → 96 % unterbricht. Ohne Clef
übernehmen einfache Regeln.

## Was du brauchst

- Eine **NVIDIA-Grafikkarte mit ~12 GB frei** für Stimme, Ohren und Clef (getestet auf einer RTX 5060 Ti 16 GB),
  ohne Clef ~7 GB. Das LLM kann irgendwo laufen – am selben Rechner, auf einem anderen oder in der Cloud.
- Python 3.12, einen OpenAI-kompatiblen LLM-Endpunkt und für Clef [llama.cpp](https://github.com/ggml-org/llama.cpp)
  ab b11719 (dessen `llama-server` hat den Endpunkt `/v1/systemone` und kennt das Entscheidungsformat von d3-mini).
- Einen Browser – und **https** fürs Mikrofon (siehe unten).

Installation, Start und alle Einstellungen: siehe die [englische README](README.md#install) – die Befehle sind dieselben.

## Optional: Phonon statt der Qwen-Stimme (CPU, Deutsch + Englisch)

`services/phonon_server.py` spricht dieselbe Schnittstelle wie die Qwen-Stimme, mit
[Phonon](https://huggingface.co/gradium/phonon-31m-en) von Gradium: etwa 7× schneller als Echtzeit auf einer 8-Kern-CPU,
erster Ton nach ~0,2 s, kein Grafikspeicher (die Qwen-Stimme braucht ~4,6 GB). Die Stimmen sind Paare aus einer
deutschen und einer englischen (Annika/Harper, Lotta/Zoey, Nele/Everly, Henrik/Preston, Leon/Trevor, Mika/Easton).
Kein Stimmklonen – die offenen Phonon-Modelle haben nur ihre Katalogstimmen. Installation:
`python -m venv .venv-phonon && .venv-phonon/bin/pip install -r services/requirements-phonon.txt`, Start:
`HF_TOKEN=hf_... .venv-phonon/bin/python services/phonon_server.py` (Port 8085), dann `FV_TTS_URL=http://127.0.0.1:8085`.

**Unterschiedliche Lizenzen:** `phonon-31m-en` ist Apache 2.0. Das deutsche Modell `phonon-40m-de` steht unter
**CC BY-NC 4.0 – nur nicht kommerziell** – und ist auf Hugging Face zugangsbeschränkt (Bedingungen annehmen, dann
`HF_TOKEN` oder `FV_HF_TOKEN_FILE` setzen). Für kommerzielle Nutzung auf Deutsch bei der Qwen-Stimme bleiben.

## Optional: apptor-tiny-tts, eine winzige deutsche Stimme (CPU, Apache 2.0)

`services/tinytts_server.py` spricht dieselbe Schnittstelle mit apptor-tiny-tts, einem deutschen Modell auf Basis von
[tiny-voca](https://github.com/n4ze3m/tiny-voca) von n4ze3m (keine offizielle tiny-voca-Version; Gewichte:
[apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts) und, mit Akzent-Regler für Österreichisch,
[apptor-tiny-tts-at](https://huggingface.co/apptor/apptor-tiny-tts-at); Browser-Demo: [tiny-tts.apptor.at](https://tiny-tts.apptor.at)):
4,5 Mio. Parameter, eine 21-MB-ONNX-Datei, etwa 40× schneller als Echtzeit auf vier CPU-Threads, keine Grafikkarte,
**kein eSpeak NG** (eigenes kleines Aussprache-Modell apptor-g2p-de). Zwei Stimmen,
`f2` (weiblich) und `m2` (männlich); die Phonon-Stimmen-IDs werden ebenfalls angenommen (weibliche -> f2, männliche
-> m2). Anders als `phonon-40m-de` stehen Modell und Code unter **Apache 2.0 – kommerzielle Nutzung erlaubt**.
Trainiert nach dem tiny-voca-Rezept auf ~12 Stunden synthetischer deutscher Sprache aus Qwen3-TTS (Texte: Common
Voice, CC0). Nur Deutsch: mit `FV_TINYTTS_EN_URL=http://127.0.0.1:8085` gehen englische Sätze an Phonons englisches
Modell. Installation:
`python -m venv .venv-tinytts && .venv-tinytts/bin/pip install -r services/requirements-tinytts.txt huggingface_hub`, Modelle: `.venv-tinytts/bin/hf download apptor/apptor-tiny-tts-at apptor-tiny-tts-at.onnx apptor-tiny-tts-at.json apptor-g2p-de.onnx apptor-g2p-de.json apptor-vocos-de.int8.onnx --local-dir models`, Start:
`FV_TINYTTS_MODEL=models/apptor-tiny-tts-at.onnx .venv-tinytts/bin/python services/tinytts_server.py` (Port 8089), dann `FV_TTS_URL=http://127.0.0.1:8089`.

**Österreichischer Akzent** (apptor-tiny-tts-at, v0.2, noch dezent): die Stimmen `f2-de`/`m2-de` sprechen Hochdeutsch,
`-at` mit Akzent 1,0, `-wien` 1,3, `-oarg` 1,6; `f2`/`m2` nehmen `FV_TINYTTS_ACCENT` (Standard 1,3). Das Formularfeld
`accent` (0 bis 1,6) überschreibt das pro Anfrage.

Die zweite Stufe, [Vocos](https://github.com/gemelo-ai/vocos) (MIT), nachtrainiert auf die beiden Stimmen, erzeugt die
Ausgabe für einen deutlich saubereren Klang neu. Sie ist automatisch an, wenn `models/apptor-vocos-de.int8.onnx` da ist
(ONNX, kein torch, ~0,1 s pro Satz); `FV_TINYTTS_VOCOS=` (leer) schaltet sie ab. Die ältere PyTorch-Version
(`=models/apptor-tiny-tts-vocos-de.pt` oder `=pretrained`) braucht torch und `services/requirements-tinytts-vocos.txt`.
Mit `FV_TINYTTS_G2P=espeak` wird statt des Aussprache-Modells das Programm espeak-ng verwendet.

## Optional: Fluid Speech Pro statt Whisper (Deutsch, Englisch, Kroatisch)

`services/fluid_stt_server.py` spricht dieselbe Schnittstelle (`/api/stt`, `/api/words`) mit
[Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) von FluidVoice (CC BY 4.0, 0,6 Mrd. Parameter,
auf Basis von NVIDIA Parakeet TDT 0.6B v3). Auf Common Voice Deutsch macht es weniger Fehler als Whisper
large-v3-turbo (Wortfehlerrate 6,4 % statt 7,9 % bei Sprechern aus Deutschland, 5,2 % statt 6,9 % mit österreichischem
Akzent) und ist viel schneller: ~30 ms pro Satz auf der Grafikkarte mit ~1,2 GB VRAM, über
[parakeet.cpp](https://github.com/mudler/parakeet.cpp) (MIT) und die q8_0-GGUF-Datei. Das Modell meldet die Sprache
nicht; der Server wählt sie anhand der erkannten Wörter aus `langs`. Nur diese drei Sprachen, kein `auto`.
Einrichtung (libparakeet.so mit `-DPARAKEET_SHARED=ON -DGGML_CUDA=ON` bauen, GGUF nach `models/`) steht in der englischen
README. Ohne Grafikkarte: `FV_FLUID_BACKEND=onnx` mit dem int8-ONNX-Export, ~0,9 s pro Satz auf vier CPU-Threads.
Fluid Speech Pro steht unter CC BY 4.0, nicht Apache: beim Weitergeben die Namensnennung (siehe NOTICE) mitliefern.

## Das Mikrofon braucht https

> ⚠️ Browser geben einer Webseite das Mikrofon **nur über https** frei – oder auf `http://localhost` /
> `http://127.0.0.1` am selben Rechner. Anders geöffnet (`http://192.168.x.x:8099`, ein Rechnername, ein anderes
> Gerät, die eingebaute Vorschau einer App) bleibt das Mikrofon aus, und die Seite sagt das.

| Wo du sie öffnest | Was tun |
|---|---|
| Normaler Browser (Chrome, Edge, Firefox, Safari) **am selben Rechner** | `http://127.0.0.1:8099/` – sonst nichts |
| Handy / anderes Gerät, **überall** (am einfachsten, echtes Zertifikat) | [Tailscale](https://tailscale.com): `tailscale serve --bg --https=8443 http://127.0.0.1:8099`, dann `https://<rechner>.<tailnet>.ts.net:8443/` – geht auch unterwegs |
| Handy / anderes Gerät **im selben WLAN** | `.env`: `FV_HOST=0.0.0.0`, `FV_CERT=self`, `FV_PASSWORD=…`; `https://<pc-ip>:8099/` öffnen und die Zertifikatswarnung einmal bestätigen |
| Du hast eine eigene Domain | ein Reverse Proxy mit Zertifikat (Caddy, nginx …) vor `127.0.0.1:8099` |

Im Auto über Bluetooth übernimmt das Auto die Echo-Unterdrückung – dort macht es am meisten Spaß.

## Grenzen

- Solange die Stimme noch rechnet, teilt sich Whisper mit ihr die Grafikkarte: eine Unterbrechung wird dann in ~1,5–3 s
  bestätigt statt ~1 s (leiser wird es trotzdem sofort).
- Einen Satz an jemand anderen im Raum („gibst du mir das Wasser?“) hielt clef-flash oft für eine Unterbrechung;
  d3-mini hatte im Test alle solchen Sätze richtig (14 von 14 Unterbrechungsfällen, clef-flash 10 von 14). Im Zweifel
  lieber einmal zu oft still als drüberreden.
- Keine Werkzeuge, Dateien oder Gedächtnis – das ist die Stimme, das Gehirn bringst du mit.

## Lizenz

Apache 2.0 (siehe [LICENSE](LICENSE)); Modelle und Bibliotheken behalten ihre eigenen, freien Lizenzen ([NOTICE](NOTICE)) –
außer dem optionalen deutschen Phonon-Modell (CC BY-NC 4.0, nur nicht kommerziell; die deutsche apptor-tiny-tts-Stimme ist
die Apache-2.0-Alternative).
Stimmen nur von Menschen klonen, die zugestimmt haben.
