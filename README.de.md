# ⚡ flash-voice

**Mit dem eigenen LLM reden – und ihm ins Wort fallen wie einem Menschen.** Ein lokaler Sprach-Stack vor jedem
OpenAI-kompatiblen Modell (Ollama, llama.cpp, vLLM, LM Studio, OpenAI, OpenRouter …).

[English](README.md)

- **Unterbrechen, das funktioniert:** Es hört die ganze Zeit zu, auch während es spricht. Sagst du etwas, wird es
  sofort leiser; war es nur „mhm“, ein Husten, das eigene Echo oder jemand anderer im Raum, redet es weiter – willst du
  wirklich etwas, hört es auf und hört dich **ab deiner ersten Silbe**.
- **Es merkt, wann du fertig bist:** Bei der ersten kurzen Pause fragt der Entscheider „ist der Gedanke fertig?“.
  „Wie wird das Wetter morgen in Wien?“ → die Antwort beginnt sofort. „Ich hätte gern, ähm …“ → es wartet.
- **Schnell:** Die Antwort wird Satz für Satz gesprochen, während das LLM noch schreibt.
- **Klein:** Der ganze Sprach-Stack läuft mit **~1,2 GB Grafikspeicher oder nur auf der CPU** – der Entscheider ist
  [flash-turn](https://huggingface.co/apptor/flash-turn), ein 73-MB-Modell, das in ~11 ms auf einem CPU-Kern antwortet.
- **Sprach-Chat für Open WebUI:** Ein `docker compose up` gibt [Open WebUI](https://github.com/open-webui/open-webui)
  Stimme und Ohren, nur CPU ([unten](#open-webui-sprach-chat-mit-einem-befehl)).
- **Lokal und frei verwendbar:** Alle Teile sind Apache 2.0, MIT oder CC BY. Der Server speichert nichts; das Gespräch
  liegt in deinem Browser.

## Wie klein es geht

flash-voice besteht aus Bausteinen, jeder ist ein eigener kleiner Dienst – du nimmst, was zu deiner Hardware passt.
Das LLM selbst ist nicht mitgerechnet: Es kann irgendwo laufen (am selben Rechner, auf einem anderen, in der Cloud).

| Baustein | Auswahl | Grafikspeicher | Hinweise |
|---|---|---|---|
| **Seite + Ohr** (immer) | flash-voice, Silero VAD | – | Python, nur CPU |
| **Entscheider** | [flash-turn](https://huggingface.co/apptor/flash-turn) (Standard) | – | Englisch + Deutsch, 73 MB, ~11 ms auf einem CPU-Kern |
| | vllm-d3-mini 4B (optional) | ~5 GB | der große |
| **Ohren** (Sprache → Text) | [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) q8_0 | **~1,2 GB** | Englisch, Deutsch, Kroatisch; ~30 ms pro Satz; im Test die wenigsten Fehler |
| | Fluid Speech Pro int8 auf der CPU | – | dasselbe Modell, ~1,4 s pro Satz auf 4 Kernen |
| | Whisper large-v3-turbo | ~1,2 GB | ~100 Sprachen, ~0,4 s pro Satz |
| **Stimme** (Text → Sprache) | **lite**: Phonon Englisch + [apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts-at) Deutsch | – | CPU, ~0,5 s pro Satz; Deutsch auch mit österreichischem Akzent |
| | Qwen3-TTS 1.7B | ~4,6 GB | 10 Sprachen, bester Klang, Stimmen klonen |

**Pakete:**

| Paket | Bausteine | Grafikspeicher | Start |
|---|---|---|---|
| **Lite** | Fluid + lite-Stimme + flash-turn | **~1,2 GB** – oder **keiner**, Ohren auf der CPU | `TTS=lite STT=fluid` |
| **Full** | Qwen3-TTS + Fluid (oder Whisper) + flash-turn | ~6 GB | `STT=fluid` |
| **Open WebUI** | Fluid (CPU) + lite-Stimme in einem Container | keiner, ~1 GB RAM | `docker compose up -d` |

Seite, Ohr und Entscheider funktionieren in jedem Paket gleich. Was Full dazu bringt: Die Qwen-Stimme klingt
natürlicher, spricht mehr Sprachen und kann Stimmen klonen.

## So funktioniert es

```
 Browser / Handy (eine Seite)                         flash-voice (Python, CPU)
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
                                                    Qwen3   4.6 GB      Whisper  1.2 GB    oder vllm-d3-mini
                                                                                           ~5 GB, oder Regeln
```

| Moment | Was passiert | Zeit |
|---|---|---|
| Du hörst auf zu reden | erste Pause (0,4 s) → Ohren + Entscheider „fertig?“ | Antwort beginnt ~1,3–1,5 s nach deinem letzten Wort, Spracherkennung inklusive |
| „Ich hätte gern, ähm …“ | Entscheider: nicht fertig → wartet bis 1,8 s Stille | |
| Du redest, während es spricht | 160 ms Stimme → es wird leiser (25 %) | ~0,2 s |
| … war „mhm“ / Echo / Geräusch | Ohren + Regeln + Entscheider → wieder laut | ~1 s |
| … war eine Unterbrechung | Entscheider: Unterbrechung → es hört auf und hört weiter zu | ~1,5 s nach deinem ersten Wort |

Der Entscheider beantwortet zwei Fragen: *ist der Gedanke fertig?* und *was tut der Mensch, während der Assistent
spricht?* (unterbricht / hört nur zu / redet mit jemand anderem). Standard ist
[flash-turn](https://huggingface.co/apptor/flash-turn), siehe [Der Entscheider](#der-entscheider). Ganz ohne
Entscheider übernehmen einfache Regeln: 1,1 s Stille beendet deinen Satz, Wörter außer „mhm“, „ja“ … unterbrechen.

## Open WebUI: Sprach-Chat mit einem Befehl

flash-voice lite ist ein kleiner Container, der [Open WebUI](https://github.com/open-webui/open-webui) Stimme und Ohren
gibt – Englisch und Deutsch, nur CPU, keine Grafikkarte, ~1 GB RAM. Er spricht die OpenAI-Audio-Schnittstelle, die
Open WebUI schon versteht: kein Plugin, nichts von Hand einzustellen.

```bash
curl -O https://raw.githubusercontent.com/ViktorsGitUser/flash-voice/main/docker/lite/docker-compose.yml
docker compose up -d          # dann http://localhost:3000 öffnen und im Chat auf das Headset-Symbol tippen
```

Die Compose-Datei startet Open WebUI gleich fertig verbunden (und mit Ollama auf deinem Rechner – `OLLAMA_BASE_URL`
ändern oder in Open WebUI eine OpenAI-kompatible API eintragen). **Open WebUI läuft schon?** Dann nur die Stimme
starten (Befehle in der [englischen README](README.md#open-webui-voice-chat-in-one-command)) und in Open WebUI unter
*Admin-Bereich → Einstellungen → Audio* für Sprache-zu-Text und Text-zu-Sprache jeweils **OpenAI** wählen, URL
`http://<host>:8090/v1`, Schlüssel `none`, Modelle `whisper-1` und `tts-1`, Stimme `m2` (oder `f2`; `m2-at` mit
österreichischem Akzent).

Gemessen auf 4 CPU-Kernen: Ein Satz ist nach ~0,3 s (Deutsch) bzw. ~0,5–0,9 s (Englisch) gesprochen und in ~1,4 s
erkannt; der Container braucht ~0,9 GB RAM, die Modelle ~1 GB Platte (beim ersten Start geladen). Der Anrufmodus von
Open WebUI hat seine eigene, einfachere Gesprächsführung; Unterbrechen und Entscheider gibt es in der Seite von
flash-voice.

## Bei dir einhängen

| Du willst … | Was du tust | Aufwand |
|---|---|---|
| **eine Stimme für dein vorhandenes LLM** | installieren, `FV_LLM_URL` in `.env` setzen (Ollama, LM Studio, llama.cpp, OpenAI …), starten, Seite öffnen | ~15 Min. |
| **Sprache in deiner eigenen App** | mit flash-voice über `WS /api/ear` und `POST /api/turn` reden ([Beschreibung](README.md#plug-it-into-your-own-setup)); `web/index.html` ist ein vollständiges Beispiel | ein Nachmittag |
| **nur einen Baustein** (z. B. nur Vorlesen oder nur Spracherkennung) | jeder Dienst ist einfaches HTTP und läuft allein | Minuten |

```bash
curl -F "text=Guten Morgen!" -F speaker=m2 http://127.0.0.1:8086/speak -o hallo.wav   # Stimme: Text -> WAV
curl -F audio=@frage.webm -F langs=de,en http://127.0.0.1:8087/api/stt                # Ohren: -> {"text","lang"}
```

Die deutsche Stimme ganz ohne Server, im Browser oder mit ein paar Zeilen Python:
[apptor-tiny-tts-at](https://huggingface.co/apptor/apptor-tiny-tts-at).
Was es (noch) nicht gibt: ein Docker-Image oder ein pip-Paket. Jeder Baustein hat seine eigene Python-Umgebung (torch
braucht nur die Qwen-Stimme); getestet unter Linux und Windows mit NVIDIA-Grafikkarten.

## Installieren und starten

Die Installationsbefehle je Baustein stehen in der [englischen README](README.md#install). Start:

```bash
TTS=lite STT=fluid scripts/start-all.sh                                 # Lite
STT=fluid scripts/start-all.sh                                          # Full (ohne STT=: Whisper)
FV_DECIDER=llm LLAMA_SERVER=/pfad/zu/llama-server scripts/start-all.sh  # mit dem großen Entscheider
# Windows: scripts\start-all.bat mit denselben Variablen (set TTS=lite usw.)
```

Seite öffnen, Mikrofon antippen, reden. Die Kugel antippen unterbricht von Hand.

## Die Bausteine im Detail

### Ohren: Fluid Speech Pro (Deutsch, Englisch, Kroatisch)

`services/fluid_stt_server.py`: [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) von
FluidVoice (CC BY 4.0, 0,6 Mrd. Parameter, auf Basis von NVIDIA Parakeet TDT 0.6B v3) über
[parakeet.cpp](https://github.com/mudler/parakeet.cpp) (MIT). Auf Common Voice Deutsch macht es weniger Fehler als
Whisper large-v3-turbo (Wortfehlerrate 6,4 % statt 7,9 % bei Sprechern aus Deutschland, 5,2 % statt 6,9 % mit
österreichischem Akzent) und ist ~10× schneller. Das Modell meldet die Sprache nicht; der Server wählt sie anhand der
erkannten Wörter aus `langs`. Nur diese drei Sprachen, kein `auto` – für andere Sprachen Whisper nehmen. Ohne
Grafikkarte: `FV_FLUID_BACKEND=onnx` mit dem int8-ONNX-Export, ~0,9 s pro Satz auf vier CPU-Threads.
Fluid Speech Pro steht unter CC BY 4.0, nicht Apache: beim Weitergeben die Namensnennung (siehe NOTICE) mitliefern.

### Stimme: apptor-tiny-tts (Deutsch, CPU, Apache 2.0)

`services/tinytts_server.py` mit apptor-tiny-tts, einem deutschen Modell auf Basis von
[tiny-voca](https://github.com/n4ze3m/tiny-voca) von n4ze3m (keine offizielle tiny-voca-Version; Gewichte:
[apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts) und, mit Akzent-Regler für Österreichisch,
[apptor-tiny-tts-at](https://huggingface.co/apptor/apptor-tiny-tts-at); Browser-Demo: [tiny-tts.apptor.at](https://tiny-tts.apptor.at)):
4,5 Mio. Parameter, eine 21-MB-ONNX-Datei, etwa 40× schneller als Echtzeit auf vier CPU-Threads, keine Grafikkarte,
**kein eSpeak NG** (eigenes kleines Aussprache-Modell apptor-g2p-de). Zwei Stimmen, `f2` (weiblich) und `m2`
(männlich). Modell und Code stehen unter **Apache 2.0 – kommerzielle Nutzung erlaubt**. Trainiert nach dem
tiny-voca-Rezept auf ~12 Stunden synthetischer deutscher Sprache aus Qwen3-TTS (Texte: Common Voice, CC0). Nur
Deutsch: mit `FV_TINYTTS_EN_URL=http://127.0.0.1:8085` gehen englische Sätze an Phonons englisches Modell.

**Österreichischer Akzent** (apptor-tiny-tts-at, v0.2, noch dezent): die Stimmen `f2-de`/`m2-de` sprechen Hochdeutsch,
`-at` mit Akzent 1,0, `-wien` 1,3, `-oarg` 1,6; `f2`/`m2` nehmen `FV_TINYTTS_ACCENT` (Standard 1,3). Das Formularfeld
`accent` (0 bis 1,6) überschreibt das pro Anfrage. Die zweite Stufe, [Vocos](https://github.com/gemelo-ai/vocos)
(MIT), nachtrainiert auf die beiden Stimmen, sorgt für einen deutlich saubereren Klang (automatisch an, wenn
`models/apptor-vocos-de.int8.onnx` da ist; `FV_TINYTTS_VOCOS=` schaltet sie ab).

### Stimme: Phonon (CPU, Deutsch + Englisch)

[Phonon](https://huggingface.co/gradium/phonon-31m-en) von Gradium: etwa 7× schneller als Echtzeit auf einer
8-Kern-CPU, erster Ton nach ~0,2 s. Die Stimmen sind Paare aus einer deutschen und einer englischen (Annika/Harper,
Lotta/Zoey, Nele/Everly, Henrik/Preston, Leon/Trevor, Mika/Easton). Kein Stimmklonen. **Unterschiedliche Lizenzen:**
`phonon-31m-en` ist Apache 2.0, das deutsche Modell `phonon-40m-de` steht unter **CC BY-NC 4.0 – nur nicht
kommerziell** – und ist auf Hugging Face zugangsbeschränkt. Für kommerzielle Nutzung auf Deutsch apptor-tiny-tts oder
die Qwen-Stimme nehmen.

### Stimme: Qwen3-TTS 1.7B (Grafikkarte)

9 eingebaute Stimmen, 10 Sprachen; mit dem `…-1.7B-Base`-Modell und `FV_VOICES=<Ordner>` eine **Stimmen-Bibliothek**:
jede Aufnahme im Ordner ist eine Stimme, neue entstehen über `POST /voices/clone` (siehe [voices/README.md](voices/README.md)).

### Der Entscheider

**flash-turn** (Standard, [apptor/flash-turn](https://huggingface.co/apptor/flash-turn), Apache 2.0): ein kleines
Textmodell für beide Fragen, Englisch und Deutsch. Basis distilbert-base-multilingual-cased (Wortschatz auf Englisch +
Deutsch gekürzt), trainiert auf ~2,8 Mio. deutschen und ~2 Mio. englischen Sätzen (ganz = fertig, nach einem Wort
abgeschnitten = nicht fertig) und ~21.000 Einwürfen, während ein Assistent spricht (von einem lokalen LLM geschrieben
und von vllm-d3-mini geprüft, dazu Anreden mit Namen). ONNX int8, 73 MB, ~11 ms pro Frage auf einem CPU-Thread.

Gemessen auf Testsätzen, die im Training nie vorkamen; vllm-d3-mini so gefragt, wie flash-voice ihn fragt:

| Test | flash-turn | vllm-d3-mini |
|---|---|---|
| Redeende, Entscheider-Testfälle (Englisch / Deutsch, je 24) | 23 / 23 | 24 / 23 |
| Einwürfe, Entscheider-Testfälle (Englisch / Deutsch, je 28) | **26 / 25** | 24 / 18 |
| Redeende, 300 Clips österreichischer Sprecher (Text der Spracherkennung) | **92,3 %** | 87,0 % |
| Redeende, unbekannte Sätze (Englisch / Deutsch) | 91,5 % / 91,0 % | 90,7 % / 90,7 % |
| Zeit pro Frage | **11 ms auf 1 CPU-Thread** | ~35 ms auf der GPU |
| Größe | **73 MB** | ~5 GB |

Längere Einwürfe (ab 4 Wörtern) unterbrechen weiterhin, außer flash-turn ist sehr sicher (≥ 80 %), dass sie jemand
anderem gelten. Ein falsches „nicht fertig“ kostet nur Zeit: das Ohr wartet dann bis 1,8 s statt 1,1 s Stille.

**vllm-d3-mini** (optional, `FV_DECIDER=llm`): ein 4B-Entscheidungsmodell in `llama-server` von llama.cpp (ab b11719),
~5 GB Grafikspeicher, ~35 ms; `scripts/start-decider.sh` startet es. Die alten Namen `FV_CLEF_URL`, `NO_CLEF`,
`CLEF_GGUF` gehen weiterhin.

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

- Solange eine Grafikkarten-Stimme noch rechnet, teilen sich die Ohren mit ihr die Grafikkarte: eine Unterbrechung wird
  dann in ~1,5–3 s bestätigt statt ~1 s (leiser wird es trotzdem sofort). Mit einer CPU-Stimme passiert das nicht.
- Ob ein Satz jemand anderem im Raum gilt, lässt sich aus den Wörtern allein oft nicht sagen („gibst du mir das
  Wasser?“). Klare Fälle (ein Name oder „Schatz“, der Fernseher, ein Sportkommentator) werden erkannt; im Zweifel
  lieber einmal zu oft still als drüberreden.
- flash-turn kann Englisch und Deutsch; in anderen Sprachen entscheiden die Pausen-Regeln.
- Keine Werkzeuge, Dateien oder Gedächtnis – das ist die Stimme, das Gehirn bringst du mit.

## Lizenz

Apache 2.0 (siehe [LICENSE](LICENSE)); Modelle und Bibliotheken behalten ihre eigenen, freien Lizenzen ([NOTICE](NOTICE)) –
außer dem optionalen deutschen Phonon-Modell (CC BY-NC 4.0, nur nicht kommerziell; die deutsche apptor-tiny-tts-Stimme ist
die Apache-2.0-Alternative); Fluid Speech Pro steht unter CC BY 4.0 (Namensnennung).
Stimmen nur von Menschen klonen, die zugestimmt haben.
