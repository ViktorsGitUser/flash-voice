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

Clef ([Cloudflare clef-flash](https://huggingface.co/Cloudflare/clef-flash)) ist kein Chat-Modell: Es bekommt einen
Text und Fragen mit festen Antworten und liefert für jede eine Wahrscheinlichkeit, in ~0,3 s. Hier werden zwei Fragen
gestellt – *ist der Gedanke fertig?* und *was tut der Mensch, während der Assistent spricht?* (unterbricht / hört nur
zu / redet mit jemand anderem). Im Test: „mhm“ → 96 % zuhören, „warte mal, stopp“ → 96 % unterbricht. Ohne Clef
übernehmen einfache Regeln.

## Was du brauchst

- Eine **NVIDIA-Grafikkarte mit ~13 GB frei** für Stimme, Ohren und Clef (getestet auf einer RTX 5060 Ti 16 GB),
  ohne Clef ~7 GB. Das LLM kann irgendwo laufen – am selben Rechner, auf einem anderen oder in der Cloud.
- Python 3.12, einen OpenAI-kompatiblen LLM-Endpunkt und für Clef [llama.cpp](https://github.com/ggml-org/llama.cpp)
  ab b11556 (dessen `llama-server` hat den Endpunkt `/v1/systemone`).
- Einen Browser – und **https** fürs Mikrofon (siehe unten).

Installation, Start und alle Einstellungen: siehe die [englische README](README.md#install) – die Befehle sind dieselben.

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
- Einen Satz an jemand anderen im Raum („gibst du mir das Wasser?“) hält Clef eher für eine Unterbrechung – lieber
  einmal zu oft still als drüberreden.
- Keine Werkzeuge, Dateien oder Gedächtnis – das ist die Stimme, das Gehirn bringst du mit.

## Lizenz

Apache 2.0 (siehe [LICENSE](LICENSE)); Modelle und Bibliotheken behalten ihre eigenen, freien Lizenzen ([NOTICE](NOTICE)).
Stimmen nur von Menschen klonen, die zugestimmt haben.
