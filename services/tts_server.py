"""flash-voice TTS service: Qwen3-TTS (Apache 2.0) through faster-qwen3-tts (MIT, static cache + CUDA graphs).

Two ways to get voices:
  * default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice - its 9 built-in speakers (serena, vivian, ono_anna, sohee,
    aiden, ryan, dylan, eric, uncle_fu), nothing else needed.
  * FV_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base + FV_VOICES=<folder>: your own voices, cloned from recordings
    (name.wav = 10-30 s of clean speech, name.txt = exactly what is said). Only with the speaker's consent.

POST /speak         form: text, speaker, language (de|en|...)       -> audio/wav
POST /speak_stream  same                                             -> NDJSON, one {"audio": base64 WAV} per ~0.7 s
GET  /voices        [{"id", "label"}]      GET /health

Run:  python services/tts_server.py      (FV_TTS_PORT 8086, FV_TTS_HOST 127.0.0.1)
On an RTX 5060 Ti the 1.7B model speaks ~2x faster than real time; ~5 GB of VRAM.
"""
import base64
import io
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse
from faster_qwen3_tts import FasterQwen3TTS

MODEL = os.environ.get("FV_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
PORT = int(os.environ.get("FV_TTS_PORT", "8086"))
HOST = os.environ.get("FV_TTS_HOST", "127.0.0.1")
VOICE_DIR = Path(os.environ.get("FV_VOICES", Path(__file__).resolve().parent.parent / "voices"))
CUSTOM = "customvoice" in MODEL.lower()
BUILT_IN = {"serena": "Serena (f)", "vivian": "Vivian (f)", "ono_anna": "Anna (f)", "sohee": "Sohee (f)",
            "aiden": "Aiden (m)", "ryan": "Ryan (m)", "dylan": "Dylan (m)", "eric": "Eric (m)", "uncle_fu": "Uncle Fu (m, older)"}
LANGUAGES = {"de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
             "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
DEFAULT = os.environ.get("FV_TTS_DEFAULT", "serena")
FRAMES_PER_S = 12        # the 12 Hz codec: one generation step per 1/12 s of sound
MAX_WAIT_S = 15          # a request that waited longer than this for the model: nobody is listening any more


def recordings() -> dict:
    """{id: (wav, text)} - the clone voices (Base model only)."""
    out = {}
    if VOICE_DIR.exists():
        for f in sorted(VOICE_DIR.glob("*.wav")):
            t = f.with_suffix(".txt")
            if t.exists():
                out[f.stem] = (str(f), t.read_text(encoding="utf-8").strip())
    return out


app = FastAPI(title="flash-voice TTS")
_lock = threading.Lock()
print(f"[tts] loading {MODEL} ...", flush=True)
_model = FasterQwen3TTS.from_pretrained(MODEL, max_seq_len=1024)
VOICES = {v: (None, None) for v in BUILT_IN} if CUSTOM else recordings()
if not VOICES:
    raise SystemExit(f"[tts] no voices: put name.wav + name.txt into {VOICE_DIR} (Base model) or use the CustomVoice model")
if DEFAULT not in VOICES:
    DEFAULT = next(iter(VOICES))


def _stream(text: str, vid: str, language: str, steps: int, temperature: float = 0.7):
    lang = LANGUAGES.get(language, "German")
    if CUSTOM:
        return _model.generate_custom_voice_streaming(text=text, speaker=vid, language=lang, temperature=temperature,
                                                      chunk_size=8, max_new_tokens=steps)
    wav, ref = VOICES[vid]
    return _model.generate_voice_clone_streaming(text=text, language=lang, ref_audio=wav, ref_text=ref,
                                                 temperature=temperature, chunk_size=8, max_new_tokens=steps)


with _lock:   # record the CUDA graphs (and read every recording once) before the first request
    for vid in (list(VOICES) if not CUSTOM else [DEFAULT]):
        list(_stream("Hallo.", vid, "de", 4 * FRAMES_PER_S))
    for _ in range(2):   # the streamed way of speaking has its own first-time cost
        list(_stream("Das Licht der Sonne sieht weiß aus, besteht aber aus allen Farben des Regenbogens.", DEFAULT, "de",
                     15 * FRAMES_PER_S))
torch.cuda.empty_cache()
print(f"[tts] ready on http://{HOST}:{PORT}, {len(VOICES)} voices, VRAM {torch.cuda.memory_reserved() / 1e9:.1f} GB",
      flush=True)


def _voice(speaker: str) -> str:
    return speaker if speaker in VOICES else DEFAULT


def _wav(audio, sr) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32).reshape(-1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# Even loudness: every sentence comes out of the model at its own level (an imitated voice up to 10 dB apart). A slow
# automatic gain per voice evens that out across pieces and sentences - stresses inside a sentence stay.
TARGET_DB, MAX_GAIN_DB, SETTLE_S = -24.0, 15.0, 1.5
_gain_db: dict = {}


def _level(audio: np.ndarray, sr: int, voice: str, first: bool) -> np.ndarray:
    n = max(1, int(0.05 * sr))
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    rms = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9) if len(frames) else np.array([])
    spoken = rms[rms > -45]
    before = _gain_db.get(voice)
    if len(spoken) >= 2:
        want = float(np.clip(TARGET_DB - np.median(spoken), -MAX_GAIN_DB, MAX_GAIN_DB))
        if first or before is None:
            before = after = want   # a new sentence: the pause before it may hide a jump
        else:
            after = before + (want - before) * (1 - np.exp(-len(audio) / sr / SETTLE_S))
    else:
        after = before if before is not None else 0.0
    if before is None:
        before = after
    _gain_db[voice] = after
    out = audio * (10 ** (np.linspace(before, after, len(audio), dtype=np.float32) / 20))
    peak = float(np.abs(out).max()) if len(out) else 0.0
    return (out / peak * 0.97 if peak > 0.97 else out).astype(np.float32)


def _max_steps(text: str) -> int:
    """At most as much sound as the text can need (slow speech ~8 characters/s, plus 2 s) - a model that does not
    stop after a short sentence would otherwise talk until its cache is full."""
    return int(min(60, len(text) / 8 + 2) * FRAMES_PER_S)


def _generate(text, speaker, language, out: queue.Queue, cancel: threading.Event, asked: float):
    vid = _voice(speaker)
    t, total, pending, sr = None, 0.0, None, 24000
    try:
        with _lock:
            if time.time() - asked > MAX_WAIT_S or cancel.is_set():
                return   # waited too long: nobody listens any more
            t = time.time()
            for chunk, sr, _info in _stream(text, vid, language, _max_steps(text)):
                if cancel.is_set():
                    break
                audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                # tiny pieces (a few ms at the end) go onto the next: Safari can hang on decoding such a crumb
                pending = np.concatenate([pending, audio]) if pending is not None else audio
                if len(pending) >= 0.12 * sr:
                    total += len(pending) / sr
                    try:
                        out.put((_level(pending, sr, vid, first=total <= len(pending) / sr), sr), timeout=3)
                    except queue.Full:
                        cancel.set()
                    pending = None
            if pending is not None and len(pending) >= 0.03 * sr and not cancel.is_set():
                total += len(pending) / sr
                out.put((_level(pending, sr, vid, first=total <= len(pending) / sr), sr), timeout=3)
            torch.cuda.empty_cache()
    except Exception as e:
        print("[tts] error:", e, flush=True)
    finally:
        out.put(None)
        if t:
            print(f"[tts] {time.time() - t:.2f}s for {total:.1f}s of speech (queued {t - asked:.1f}s)", flush=True)


def _pieces(text, speaker, language):
    """The pieces as they come; when the caller stops reading (disconnect), the generation stops too."""
    out, cancel = queue.Queue(maxsize=8), threading.Event()
    threading.Thread(target=_generate, args=(text, speaker, language, out, cancel, time.time()), daemon=True).start()
    try:
        while (item := out.get()) is not None:
            yield item
    finally:
        cancel.set()


@app.post("/speak")
def speak(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de")):
    parts = list(_pieces(text, speaker, language))
    if not parts:
        return JSONResponse({"error": "no speech made"}, status_code=503)
    return Response(_wav(np.concatenate([a for a, _ in parts]), parts[0][1]), media_type="audio/wav")


@app.post("/speak_stream")
def speak_stream(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de")):
    def lines():
        for audio, sr in _pieces(text, speaker, language):
            yield json.dumps({"audio": base64.b64encode(_wav(audio, sr)).decode()}) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/voices")
def voices():
    return [{"id": v, "label": BUILT_IN.get(v, v.replace("_", " ").title())} for v in VOICES]


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "voices": len(VOICES)}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
