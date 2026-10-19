"""flash-voice STT service: Whisper large-v3-turbo (MIT) through faster-whisper (MIT, CTranslate2).

POST /api/stt  form: audio (any format ffmpeg/PyAV reads), langs ("de,en" - Whisper picks the likeliest of these;
               short clips are easily misdetected otherwise; "auto": any language - "auto,de,en" falls back to de/en when
               Whisper is unsure; bs/sr count as hr), fast ("1": greedy, for the ear's quick checks)
               -> {"text", "lang"}
POST /api/words (also /api/woerter)  form: audio, lang, text (optional hint) -> {"words": [{"w", "start", "end"}]}
GET  /health (also /api/health)

Run:  python services/stt_server.py     (FV_STT_PORT 8087, FV_STT_HOST 127.0.0.1, FV_STT_MODEL large-v3-turbo,
      FV_STT_COMPUTE int8_float16 - about half the memory of float16, practically the same recognition; ~1.5 GB)
Nothing is stored: the audio is only held in memory while it is recognised.
"""
import glob
import io
import os
import sys
import threading

if sys.platform == "win32":   # the CUDA 12 / cuDNN 9 DLLs from the nvidia-* pip packages, before ctranslate2 loads
    for _d in glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")):
        os.add_dll_directory(_d)
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from faster_whisper import WhisperModel, decode_audio  # noqa: E402

PORT = int(os.environ.get("FV_STT_PORT", "8087"))
HOST = os.environ.get("FV_STT_HOST", "127.0.0.1")
MODEL = os.environ.get("FV_STT_MODEL", "large-v3-turbo")
COMPUTE = os.environ.get("FV_STT_COMPUTE", "int8_float16")
LANGS = tuple(os.environ.get("FV_STT_LANGS", "en,de").split(","))

print(f"[stt] loading Whisper {MODEL} ({COMPUTE}) ...", flush=True)
_stt = WhisperModel(MODEL, device=os.environ.get("FV_STT_DEVICE", "cuda"), compute_type=COMPUTE, cpu_threads=1, num_workers=1)
_lock = threading.Lock()
list(_stt.transcribe(np.zeros(16000, dtype=np.float32), language="de", beam_size=1)[0])   # first run compiles GPU code
app = FastAPI(title="flash-voice STT")


# Croatian is often heard as Bosnian or Serbian (the same language in practice) - written as Croatian (Latin script)
SAME = {"bs": "hr", "sr": "hr"}
SURE = 0.5


def pick_language(audio, hints: tuple = ()) -> str:
    """The language spoken, from all of Whisper's. Only when Whisper is unsure (a short "ja", noise) the likeliest of
    `hints`. Never a forced wrong language: Croatian restricted to "de,en,hr" lost to English as its probability was
    split over bs/sr/hr - and Whisper, told "English", translated it."""
    probs = {}
    for code, p in _stt.detect_language(audio)[2]:
        probs[SAME.get(code, code)] = probs.get(SAME.get(code, code), 0.0) + p
    best = max(probs, key=probs.get) if probs else "en"
    if probs.get(best, 0) >= SURE or not hints:
        return best
    hint = max(hints, key=lambda h: probs.get(h, 0.0))
    # a hint only when it is a close second - never force a language Whisper hardly hears (it would translate)
    return hint if probs.get(hint, 0.0) >= 0.6 * probs[best] else best


def transcribe(data: bytes, langs: tuple = LANGS, fast: bool = False) -> tuple[str, str]:
    audio = decode_audio(io.BytesIO(data), sampling_rate=16000)
    if langs and langs[0] == "auto":   # any language Whisper knows, the rest of langs as a fallback when unsure
        with _lock:
            lang = pick_language(audio, langs[1:])
            segments, _ = _stt.transcribe(audio, language=lang, beam_size=1 if fast else 5, vad_filter=True)
            return "".join(seg.text for seg in segments).strip(), lang
    with _lock:
        try:
            if len(langs) == 1:
                lang = langs[0]
            else:
                probs = dict(_stt.detect_language(audio)[2])
                lang = max(langs, key=lambda l: probs.get(l, 0.0))
        except Exception:
            lang = langs[0] if langs else "de"
        segments, _ = _stt.transcribe(audio, language=lang, beam_size=1 if fast else 5, vad_filter=True)
        return "".join(seg.text for seg in segments).strip(), lang


@app.post("/api/stt")
def stt(audio: UploadFile = File(...), langs: str = Form(""), fast: str = Form("")):
    want = tuple(l.strip() for l in langs.split(",") if l.strip()) or LANGS
    text, lang = transcribe(audio.file.read(), want, fast == "1")
    return {"text": text, "lang": lang}


@app.post("/api/woerter")
@app.post("/api/words")
def words(audio: UploadFile = File(...), lang: str = Form("de"), text: str = Form("")):
    """When each word is said (word-by-word subtitles): -> {"words": [{"w", "start", "end"}]}. `text`: what is said,
    if known (a hint: names and brands come out right)."""
    data = decode_audio(io.BytesIO(audio.file.read()), sampling_rate=16000)
    with _lock:
        segments, _ = _stt.transcribe(data, language=lang or "de", beam_size=5, word_timestamps=True,
                                      vad_filter=False, initial_prompt=(text or None) and text[:400])
        out = [{"w": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
               for seg in segments for w in (seg.words or []) if w.word.strip()]
    return {"words": out}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL, "compute": COMPUTE}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
