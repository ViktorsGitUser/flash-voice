"""flash-voice STT service: Whisper large-v3-turbo (MIT) through faster-whisper (MIT, CTranslate2).

POST /api/stt  form: audio (any format ffmpeg/PyAV reads), langs ("de,en" - Whisper picks the likeliest of these;
               short clips are easily misdetected otherwise), fast ("1": greedy, for the ear's quick checks)
               -> {"text", "lang"}
GET  /health

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
LANGS = tuple(os.environ.get("FV_STT_LANGS", "de,en").split(","))

print(f"[stt] loading Whisper {MODEL} ({COMPUTE}) ...", flush=True)
_stt = WhisperModel(MODEL, device=os.environ.get("FV_STT_DEVICE", "cuda"), compute_type=COMPUTE, cpu_threads=1, num_workers=1)
_lock = threading.Lock()
list(_stt.transcribe(np.zeros(16000, dtype=np.float32), language="de", beam_size=1)[0])   # first run compiles GPU code
app = FastAPI(title="flash-voice STT")


def transcribe(data: bytes, langs: tuple = LANGS, fast: bool = False) -> tuple[str, str]:
    audio = decode_audio(io.BytesIO(data), sampling_rate=16000)
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


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
