"""flash-voice: a fast CPU voice with Phonon (Gradium) - the same HTTP API as tts_server.py, German and English.

Phonon is a small flow-matching TTS (~31-40M parameters) that runs on the CPU: about 7x faster than real time on an
8-core desktop, the first sound after ~0.05 s, ~300 MB RAM per language, no GPU memory at all. It has no voice
cloning in the open models - only its catalog voices; for your own cloned voices keep tts_server.py (Qwen3-TTS).

Voices here are PAIRS: one id, a German and an English catalog voice of a similar kind, so the voice stays itself
when the language changes in a conversation.

Licenses: phonon-31m-en Apache 2.0; phonon-40m-de CC BY-NC 4.0 (non-commercial only) and gated on Hugging Face
(accept the terms, then set HF_TOKEN or FV_HF_TOKEN_FILE). Runtime `ptts` MIT OR Apache-2.0.

POST /speak         form: text, speaker, language (de|en) -> audio/wav
POST /speak_stream  form: text, speaker, language -> NDJSON lines {"audio": base64 WAV} of ~0.5 s each
GET  /voices, /speakers, /health

Run:  python services/phonon_server.py   (FV_PHONON_PORT 8085, FV_PHONON_QUANT q8, FV_PHONON_THREADS 4)
"""
import base64
import io
import json
import os
import queue
import threading

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse

PORT = int(os.environ.get("FV_PHONON_PORT", "8085"))
HOST = os.environ.get("FV_PHONON_HOST", "127.0.0.1")
QUANT = os.environ.get("FV_PHONON_QUANT", "q8")
THREADS = int(os.environ.get("FV_PHONON_THREADS", "4"))   # 16 by default: took the CPU from Strata (its experts)
MODELS = {"de": os.environ.get("FV_PHONON_DE", "hf://gradium/phonon-40m-de"),
          "en": os.environ.get("FV_PHONON_EN", "hf://gradium/phonon-31m-en")}
SR = 24000
PIECE_S = 0.5   # stream pieces of about this length
if not os.environ.get("HF_TOKEN") and os.environ.get("FV_HF_TOKEN_FILE"):
    try:
        os.environ["HF_TOKEN"] = open(os.environ["FV_HF_TOKEN_FILE"]).read().strip()
    except OSError:
        pass

# id -> (label, German catalog voice, English catalog voice, description)
PAIRS = {
    "annika": ("Annika", "Annika", "Harper", "freundlich, auf den Punkt"),
    "lotta": ("Lotta", "Lotta", "Zoey", "lebhaft, nah"),
    "nele": ("Nele", "Nele", "Everly", "weich, ruhig"),
    "henrik": ("Henrik", "Henrik", "Preston", "entspannt, warm"),
    "leon": ("Leon", "Leon", "Trevor", "hell, einladend"),
    "mika": ("Mika", "Mika", "Easton", "knackig, kräftig"),
}
DEFAULT = os.environ.get("FV_PHONON_DEFAULT", "annika")

import ptts  # noqa: E402

ptts.set_num_threads(THREADS)

print(f"[phonon] loading {', '.join(MODELS.values())} ({QUANT}, {THREADS} threads) ...", flush=True)
_tts = {lang: ptts.TTS(m, lang=lang, quant=QUANT) for lang, m in MODELS.items()}
_lock = threading.Lock()   # one sentence at a time (the CPU threads are shared)
for _lang, _t in _tts.items():   # every voice once: its first use loads it from the catalog (~1 s, measured)
    for _p in PAIRS.values():
        list(_t.stream("Hallo." if _lang == "de" else "Hello.", voice=_p[1 if _lang == "de" else 2]))
app = FastAPI(title="flash-voice Phonon")


def _voice(speaker: str, lang: str) -> str:
    pair = PAIRS.get((speaker or "").lower()) or PAIRS[DEFAULT]
    return pair[1] if lang == "de" else pair[2]


def _lang(language: str) -> str:
    return "en" if (language or "").lower().startswith("en") else "de"


def _wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32).reshape(-1), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _chunks(text: str, speaker: str, language: str):
    """The sentence's audio as it is made. It is made in a thread of its own that holds the lock only while it
    makes it: when the caller stops reading (Flash Chat cut the speech off, another voice was chosen) the thread
    stops at the next piece and lets the lock go (22.10.2026: the lock was held by an abandoned stream - every
    later sentence waited, the voice "stuttered" and then stopped)."""
    lang = _lang(language)
    out, stop = queue.Queue(), threading.Event()

    def work():
        if not _lock.acquire(timeout=60):
            out.put(RuntimeError("voice busy"))
            out.put(None)
            return
        try:
            for c in _tts[lang].stream(text, voice=_voice(speaker, lang)):
                if stop.is_set():
                    break
                out.put(np.asarray(c, dtype=np.float32).reshape(-1))
        except Exception as e:
            out.put(e)
        finally:
            _lock.release()
            out.put(None)
    threading.Thread(target=work, daemon=True).start()
    try:
        while (item := out.get(timeout=60)) is not None:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()


@app.post("/speak")
def speak(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de"), instruct: str = Form(""),
          mix: str = Form(""), temperature: float = Form(0.7)):
    parts = list(_chunks(text, speaker, language))
    if not parts:
        return JSONResponse({"error": "no speech made"}, status_code=503)
    return Response(_wav(np.concatenate(parts)), media_type="audio/wav")


@app.post("/speak_stream")
def speak_stream(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de"),
                 instruct: str = Form(""), mix: str = Form(""), temperature: float = Form(0.7)):
    def lines():
        buf = []
        for c in _chunks(text, speaker, language):
            buf.append(c)
            if sum(len(b) for b in buf) >= PIECE_S * SR:
                yield json.dumps({"audio": base64.b64encode(_wav(np.concatenate(buf))).decode()}) + "\n"
                buf = []
        if buf:
            yield json.dumps({"audio": base64.b64encode(_wav(np.concatenate(buf))).decode()}) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/speakers")
def speakers():
    return {vid: p[0] for vid, p in PAIRS.items()}


@app.get("/voices")
def voices():
    return [{"id": vid, "label": f"{p[0]} (Phonon)", "description": f"{p[3]} – Deutsch {p[1]}, Englisch {p[2]}",
             "builtin": True} for vid, p in PAIRS.items()]


@app.get("/health")
def health():
    return {"ok": True, "engine": "phonon", "languages": list(_tts)}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
