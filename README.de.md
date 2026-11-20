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
- **Lokal und frei verwendbar:** Alle Teile sind Apache 2.0, MIT oder CC BY. Der Server speichert nichts; das Gespräch
  liegt in deinem Browser.

## Wie klein es geht

flash-voice besteht aus Bausteinen, jeder ist ein eigener kleiner Dienst – du nimmst, was zu deiner Hardware passt.
Das LLM selbst ist nicht mitgerechnet: Es kann irgendwo laufen (am selben Rechner, auf einem anderen, in der Cloud).

| Baustein | Auswahl | Grafikspeicher | Hinweise |
|---|---|---|---|
| **Seite + Ohr** (immer) | flash-voice, Silero VAD | – | Python, nur CPU |
| **Ohren** (Sprache → Text) | [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro) q8_0 | **~1,2 GB** | Deutsch, Englisch, Kroatisch; ~30 ms pro Satz; im Test die wenigsten Fehler |
| | Fluid Speech Pro int8 auf der CPU | – | dasselbe Modell, ~0,9 s pro Satz |
| | Whisper large-v3-turbo | ~1,2 GB | ~100 Sprachen, ~0,4 s pro Satz |
| **Stimme** (Text → Sprache) | [apptor-tiny-tts](https://huggingface.co/apptor/apptor-tiny-tts-at) | – | Deutsch (auch mit österreichischem Akzent), 21 MB, CPU, ~40× Echtzeit |
| | Phonon | – | Deutsch + Englisch, CPU (deutsches Modell nur nicht kommerziell) |
| | Qwen3-TTS 1.7B | ~4,6 GB | 10 Sprachen, bester Klang, Stimmen klonen |
| **Entscheider** (optional) | vllm-d3-mini 4B | ~5 GB | erkennt „fertig?“ und „Unterbrechung oder nur mhm?“; ohne ihn entscheiden einfache Regeln |

**Pakete:**

| Paket | Bausteine | Grafikspeicher | Start |
|---|---|---|---|
| **Tiny** | Fluid + apptor-tiny-tts, Regeln | **~1,2 GB** (oder keiner, Ohren auf der CPU) | `TTS=tiny STT=fluid NO_DECIDER=1` |
| **Smart** | Tiny + Entscheider | ~6 GB | `TTS=tiny STT=fluid` |
| **Full** | Qwen3-TTS + Fluid oder Whisper + Entscheider | ~11 GB | `STT=fluid` (oder nichts: Whisper) |

Seite und Ohr funktionieren in jedem Paket gleich. Was die größeren bringen: Der Entscheider wartet bei „Ich hätte
gern, ähm …“ und hört nicht auf bei „mhm“ oder einem Satz an jemand anderen im Raum; die Qwen-Stimme klingt
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
                                                    tiny-tts    CPU     Fluid    1.2 GB    vllm-d3-mini 5 GB
                                                    Phonon      CPU     Whisper  1.2 GB    optional, sonst
                                                    Qwen3   4.6 GB                         Regeln
```

| Moment | Was passiert | Zeit |
|---|---|---|
| Du hörst auf zu reden | erste Pause (0,4 s) → Ohren + Entscheider „fertig?“ | Antwort beginnt ~1,3–1,5 s nach deinem letzten Wort, Spracherkennung inklusive |
| „Ich hätte gern, ähm …“ | Entscheider: nicht fertig → wartet bis 1,8 s Stille | |
| Du redest, während es spricht | 160 ms Stimme → es wird leiser (25 %) | ~0,2 s |
| … war „mhm“ / Echo / Geräusch | Ohren + Regeln + Entscheider → wieder laut | ~1 s |
| … war eine Unterbrechung | Entscheider: Unterbrechung → es hört auf und hört weiter zu | ~1,5 s nach deinem ersten Wort |

Der Entscheider ([vllm-d3-mini](https://huggingface.co/vllm-sr/d3-mini), Apache 2.0, 4B; bis November 2026
[Cloudflare clef-flash](https://huggingface.co/Cloudflare/clef-flash), 9B) ist kein Chat-Modell: Er bekommt einen
Text und Fragen mit festen Antworten und liefert für jede eine Wahrscheinlichkeit, in ~35 ms auf der GPU. Zwei Fragen
werden gestellt – *ist der Gedanke fertig?* und *was tut der Mensch, während der Assistent spricht?* (unterbricht /
hört nur zu / redet mit jemand anderem). Im Test: „mhm“ → 96 % zuhören, „warte mal, stopp“ → 96 % unterbricht. Ohne
Entscheider übernehmen einfache Regeln: 1,1 s Stille beendet deinen Satz, Wörter außer „mhm“, „ja“ … unterbrechen.

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
TTS=tiny STT=fluid NO_DECIDER=1 scripts/start-all.sh                       # Tiny
TTS=tiny STT=fluid LLAMA_SERVER=/pfad/zu/llama-server scripts/start-all.sh # Smart
STT=fluid LLAMA_SERVER=/pfad/zu/llama-server scripts/start-all.sh          # Full (ohne STT=: Whisper)
# Windows: scripts\start-all.bat mit denselben Variablen (set TTS=tiny usw.)
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

`scripts/start-decider.sh` startet vllm-d3-mini in `llama-server` von llama.cpp (ab b11719: hat den Endpunkt
`/v1/systemone` und kennt das Entscheidungsformat von d3-mini), nur Text, ~5 GB Grafikspeicher. flash-voice fragt ihn
unter `FV_DECIDER_URL` (leer = nur Regeln; die alten Namen `FV_CLEF_URL`, `NO_CLEF`, `CLEF_GGUF` gehen weiterhin).

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
- Einen Satz an jemand anderen im Raum („gibst du mir das Wasser?“) hielt clef-flash oft für eine Unterbrechung;
  d3-mini hatte im Test alle solchen Sätze richtig (14 von 14 Unterbrechungsfällen, clef-flash 10 von 14). Im Zweifel
  lieber einmal zu oft still als drüberreden. Ohne Entscheider ist das die Schwachstelle der Regeln.
- Keine Werkzeuge, Dateien oder Gedächtnis – das ist die Stimme, das Gehirn bringst du mit.

## Lizenz

Apache 2.0 (siehe [LICENSE](LICENSE)); Modelle und Bibliotheken behalten ihre eigenen, freien Lizenzen ([NOTICE](NOTICE)) –
außer dem optionalen deutschen Phonon-Modell (CC BY-NC 4.0, nur nicht kommerziell; die deutsche apptor-tiny-tts-Stimme ist
die Apache-2.0-Alternative); Fluid Speech Pro steht unter CC BY 4.0 (Namensnennung).
Stimmen nur von Menschen klonen, die zugestimmt haben.
