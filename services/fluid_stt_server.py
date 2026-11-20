"""flash-voice STT service, alternative to stt_server.py: Fluid Speech Pro (FluidVoice, CC BY 4.0; built on NVIDIA
Parakeet TDT 0.6B v3 and Moondream Parakeet Ultra), German / English / Croatian. Same API as stt_server.py, so the ear
can switch by port or by starting this one instead.

POST /api/stt  form: audio (any format PyAV reads), langs ("de,en,hr": the language is guessed from the recognised
               text among these - the model itself needs no language), fast (ignored, always greedy) -> {"text", "lang"}
POST /api/words (also /api/woerter)  form: audio, lang, text (ignored) -> {"words": [{"w", "start", "end"}]}
GET  /health (also /api/health)

Two backends (FV_FLUID_BACKEND):
  gguf (default)  parakeet.cpp (MIT, github.com/mudler/parakeet.cpp) through its C API, GPU: the q8_0 GGUF file,
                  ~1.2 GB VRAM, ~30 ms per sentence. Needs libparakeet.so (cmake -DPARAKEET_SHARED=ON -DGGML_CUDA=ON).
  onnx            onnx-asr on the CPU: the int8 ONNX files (~0.9 s per sentence on four threads, no GPU).
On Common Voice German (200 clips from Germany, 184 with Austrian accent) the word error rate was 6.4 % / 5.2 %
(Whisper large-v3-turbo: 7.9 % / 6.9 %).

Run:  python services/fluid_stt_server.py   (FV_STT_PORT 8087, FV_STT_HOST 127.0.0.1, FV_STT_LANGS en,de,
      FV_FLUID_MODEL models/fluid-speech-pro-q8_0.gguf (gguf) or models/fluid-speech-pro-onnx (onnx),
      FV_FLUID_LIB libparakeet.so (path, or found through LD_LIBRARY_PATH), FV_FLUID_THREADS 4 (onnx))
Nothing is stored: the audio is only held in memory while it is recognised.
"""
import atexit
import ctypes
import io
import json
import os
import re
import threading
from pathlib import Path

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("FV_STT_PORT", "8087"))
HOST = os.environ.get("FV_STT_HOST", "127.0.0.1")
LANGS = tuple(os.environ.get("FV_STT_LANGS", "en,de").split(","))
BACKEND = os.environ.get("FV_FLUID_BACKEND", "gguf")
MODEL = os.environ.get("FV_FLUID_MODEL", str(ROOT / "models" / ("fluid-speech-pro-q8_0.gguf" if BACKEND == "gguf"
                                                                  else "fluid-speech-pro-onnx")))
SR = 16000


def decode(data: bytes) -> np.ndarray:
    """Any audio file -> 16 kHz mono float32."""
    out = []
    with av.open(io.BytesIO(data)) as f:
        rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for frame in f.decode(audio=0):
            out += [x.to_ndarray().reshape(-1) for x in rs.resample(frame)]
        out += [x.to_ndarray().reshape(-1) for x in rs.resample(None)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


class Gguf:
    """parakeet.cpp C API: one context, kept loaded."""

    def __init__(self, model: str):
        lib = ctypes.CDLL(os.environ.get("FV_FLUID_LIB", "libparakeet.so"))
        lib.parakeet_capi_load.restype = ctypes.c_void_p
        lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
        lib.parakeet_capi_load_error.restype = ctypes.c_char_p
        lib.parakeet_capi_free.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
        lib.parakeet_capi_transcribe_pcm_batch_json.restype = ctypes.c_void_p
        lib.parakeet_capi_transcribe_pcm_batch_json.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
            ctypes.c_int]
        self.lib = lib
        self.ctx = lib.parakeet_capi_load(model.encode())
        if not self.ctx:
            raise RuntimeError(f"parakeet.cpp could not load {model}: {lib.parakeet_capi_load_error().decode()}")
        atexit.register(self.close)

    def close(self):
        if self.ctx:
            self.lib.parakeet_capi_free(self.ctx)
            self.ctx = None

    def run(self, audio: np.ndarray) -> dict:
        audio = np.ascontiguousarray(audio, np.float32)
        n = (ctypes.c_int * 1)(len(audio))
        p = self.lib.parakeet_capi_transcribe_pcm_batch_json(
            self.ctx, audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, 1, SR, 0)
        if not p:
            raise RuntimeError("parakeet.cpp transcription failed")
        try:
            return json.loads(ctypes.string_at(p).decode("utf-8"))[0]
        finally:
            self.lib.parakeet_capi_free_string(p)


class Onnx:
    """onnx-asr (MIT) with the int8 ONNX export, CPU."""

    def __init__(self, model: str):
        import onnx_asr
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("FV_FLUID_THREADS", "4"))
        self.m = onnx_asr.load_model("nemo-conformer-tdt", model, quantization="int8", sess_options=so,
                                     providers=["CPUExecutionProvider"])
        self.tm = self.m.with_timestamps()

    def run(self, audio: np.ndarray) -> dict:
        r = self.tm.recognize(audio, sample_rate=SR)
        words, cur = [], None
        for tok, t in zip(r.tokens or [], r.timestamps or []):   # SentencePiece pieces -> words
            if tok.startswith(" ") or cur is None:
                cur = {"w": tok.strip(), "start": t, "end": t}
                words.append(cur)
            else:
                cur["w"] += tok
                cur["end"] = t
        return {"text": r.text, "words": [w for w in words if w["w"]]}


print(f"[stt] loading Fluid Speech Pro ({BACKEND}: {MODEL}) ...", flush=True)
_asr = Gguf(MODEL) if BACKEND == "gguf" else Onnx(MODEL)
_lock = threading.Lock()
_asr.run(np.zeros(SR, np.float32))   # warm up (CUDA graphs, buffers)
app = FastAPI(title="flash-voice STT (Fluid Speech Pro)")

# The model does not say which language it heard: frequent short words decide (only the languages asked for).
COMMON = {
    "de": "der die das und ist nicht ich du er sie es wir ihr ein eine zu mit auf für von den dem im ja nein auch was "
          "wie wo kann bitte danke noch schon aber oder wenn dass hab habe sind war mir mich dir dich hallo heute",
    "en": "the and is are not you he she it we they a an to with on for of in yes no also what how where can please "
          "thanks still already but or if that have has was me my your hello today i this do does",
    "hr": "je i u na da se ne to su sam si smo što kako gdje može molim hvala još već ali ili ako imam bio bila "
          "mene meni tebi bok danas ovo ja ti mi vi oni",
}
COMMON = {k: set(v.split()) for k, v in COMMON.items()}


def guess_lang(text: str, langs: tuple) -> str:
    cands = [l for l in langs if l in COMMON] or list(langs) or ["de"]
    words = re.findall(r"\w+", text.lower())
    score = {l: sum(w in COMMON.get(l, ()) for w in words) for l in cands}
    if "de" in score:
        score["de"] += 2 * sum(c in "äöüß" for c in text.lower())
    if "hr" in score:
        score["hr"] += 2 * sum(c in "čćšžđ" for c in text.lower())
    best = max(cands, key=lambda l: (score.get(l, 0), -cands.index(l)))
    return best


def transcribe(data: bytes, langs: tuple = LANGS) -> tuple[str, str]:
    audio = decode(data)
    if len(audio) < SR // 10:
        return "", (langs[0] if langs and langs[0] != "auto" else "de")
    with _lock:
        text = _asr.run(audio)["text"].strip()
    return text, guess_lang(text, tuple(l for l in langs if l != "auto"))


@app.post("/api/stt")
def stt(audio: UploadFile = File(...), langs: str = Form(""), fast: str = Form("")):
    want = tuple(l.strip() for l in langs.split(",") if l.strip()) or LANGS
    text, lang = transcribe(audio.file.read(), want)
    return {"text": text, "lang": lang}


@app.post("/api/woerter")
@app.post("/api/words")
def words(audio: UploadFile = File(...), lang: str = Form("de"), text: str = Form("")):
    data = decode(audio.file.read())
    with _lock:
        r = _asr.run(data)
    return {"words": [{"w": w["w"], "start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3)}
                      for w in r.get("words", []) if w["w"].strip()]}


@app.get("/health")
@app.get("/api/health")
def health():
    return {"ok": True, "model": "fluid-speech-pro", "backend": BACKEND}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
