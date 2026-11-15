"""flash-voice: apptor-tiny-tts, a tiny German CPU voice built on tiny-voca - the same HTTP API as tts_server.py and phonon_server.py.

tiny-voca (github.com/n4ze3m/tiny-voca) is a 4.5M-parameter TTS with its own waveform generator: ~18 MB, no GPU,
many times faster than real time on a few CPU threads. apptor-tiny-tts is a German model (not an official tiny-voca release) trained with that recipe on
synthetic speech from Qwen3-TTS (two voices, F2 female and M2 male, 24 kHz). Text -> phonemes with eSpeak NG "de".

License: code and German weights Apache 2.0 (unlike phonon-40m-de, which is CC BY-NC 4.0 - non-commercial only).
Runtime needs the espeak-ng program (GPL-3.0, called as a separate process, not bundled).

The model is German only. An English request is passed on to FV_TINYTTS_EN_URL (e.g. the Phonon server, whose
English model is Apache 2.0) if that is set; otherwise the German voice reads it.

POST /speak         form: text, speaker, language (de|en) -> audio/wav
POST /speak_stream  form: text, speaker, language -> NDJSON lines {"audio": base64 WAV} of ~0.5 s each
GET  /voices, /speakers, /health

Optional second stage (FV_TINYTTS_VOCOS): the waveform of the tiny-voca model is re-synthesised by Vocos (MIT, 13.5M
parameters, CPU) from its mel spectrogram - this removes most of the buzzy sound of the tiny source-filter generator.
FV_TINYTTS_VOCOS=models/apptor-tiny-tts-vocos-de.pt uses the version fine-tuned on the two German voices; =pretrained the generic
charactr/vocos-mel-24khz. Needs torch (CPU build is enough) and the vocos package (requirements-tinytts-vocos.txt).

Run:  python services/tinytts_server.py   (FV_TINYTTS_PORT 8089, FV_TINYTTS_MODEL models/apptor-tiny-tts-de.onnx,
      FV_TINYTTS_THREADS 4, FV_TINYTTS_SPEED 1.0, FV_TINYTTS_VOCOS models/apptor-tiny-tts-vocos-de.pt)
"""
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinyvoca.text import phonemize  # noqa: E402  (vendored tiny-voca front end, German)

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("FV_TINYTTS_PORT", "8089"))
HOST = os.environ.get("FV_TINYTTS_HOST", "127.0.0.1")
MODEL = Path(os.environ.get("FV_TINYTTS_MODEL", ROOT / "models" / "apptor-tiny-tts-de.onnx"))
THREADS = int(os.environ.get("FV_TINYTTS_THREADS", "4"))   # Strata needs the CPU too
SPEED = float(os.environ.get("FV_TINYTTS_SPEED", "1.0"))
EN_URL = os.environ.get("FV_TINYTTS_EN_URL", "")         # e.g. http://127.0.0.1:8085 (Phonon, English)
VOCOS = os.environ.get("FV_TINYTTS_VOCOS", "")             # "", "pretrained" or a fine-tuned state dict (.pt)
PIECE_S = 0.5
GAP_S = 0.12   # silence between the parts of a long sentence

META = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))
SR = META["sample_rate"]
MAX_TOKENS = META["max_tokens"]

# id -> (label, model voice, description). The Phonon ids are kept, so a client that chose a Phonon voice still
# gets one of the same kind.
VOICES = {
    "f2": ("Tiny F2", "F2", "weiblich, klar"),
    "m2": ("Tiny M2", "M2", "männlich, ruhig"),
}
ALIASES = {"annika": "f2", "lotta": "f2", "nele": "f2", "henrik": "m2", "leon": "m2", "mika": "m2"}
DEFAULT = os.environ.get("FV_TINYTTS_DEFAULT", "f2")

options = ort.SessionOptions()
options.intra_op_num_threads = THREADS
options.inter_op_num_threads = 1
print(f"[tiny-tts] loading {MODEL} ({THREADS} threads) ...", flush=True)
_session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
_vocos = None
if VOCOS:
    import torch
    from vocos import Vocos
    torch.set_num_threads(THREADS)
    _vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
    if VOCOS != "pretrained":
        path = Path(VOCOS) if Path(VOCOS).is_absolute() else ROOT / VOCOS
        _vocos.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    print(f"[tiny-tts] second stage: Vocos ({VOCOS})", flush=True)
app = FastAPI(title="flash-voice apptor-tiny-tts")


def _voice_index(speaker: str) -> int:
    vid = (speaker or "").lower()
    vid = ALIASES.get(vid, vid)
    return META["voices"].index(VOICES.get(vid, VOICES[DEFAULT])[1])


def _parts(text: str) -> list[str]:
    """Pieces short enough for one utterance (<= MAX_TOKENS phonemes): sentences, then clauses, then words."""
    out = []
    for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if not sentence:
            continue
        try:
            if len(phonemize(sentence).ids) <= MAX_TOKENS:
                out.append(sentence)
                continue
        except ValueError:
            continue   # nothing to pronounce
        pieces = re.split(r"(?<=[,;:–])\s+", sentence)
        if len(pieces) == 1:   # no clause boundary: cut in the middle of the words
            words = sentence.split()
            pieces = [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
        for p in pieces:
            out.extend(_parts(p))
    return out


def _synth(text: str, voice: int, speed: float = SPEED) -> np.ndarray:
    phones = phonemize(text)
    audio, _ = _session.run(None, {
        "tokens": np.array([phones.ids], dtype=np.int64),
        "stress": np.array([phones.stress], dtype=np.int64),
        "voice": np.array([voice], dtype=np.int64),
        "speed": np.array([speed], dtype=np.float32),
        "pitch_shift": np.array([0.0], dtype=np.float32),
        "pitch_range": np.array([1.0], dtype=np.float32),
    })
    audio = audio[0].astype(np.float32)
    if _vocos is not None:
        with torch.inference_mode():
            x = torch.from_numpy(audio)[None]
            audio = _vocos.decode(_vocos.feature_extractor(x))[0, :len(audio)].numpy()
    peak = float(np.abs(audio).max())
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return audio.astype(np.float32)


def _chunks(text: str, speaker: str):
    voice = _voice_index(speaker)
    for i, part in enumerate(_parts(text)):
        if i:
            yield np.zeros(int(GAP_S * SR), dtype=np.float32)
        yield _synth(part, voice)


def _wav(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32).reshape(-1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _english(language: str) -> bool:
    return bool(EN_URL) and (language or "").lower().startswith("en")


def _forward(path: str, form: dict):
    import httpx
    return httpx.post(EN_URL.rstrip("/") + path, data=form, timeout=120)


@app.post("/speak")
def speak(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de"), instruct: str = Form(""),
          mix: str = Form(""), temperature: float = Form(0.7)):
    if _english(language):
        r = _forward("/speak", {"text": text, "speaker": speaker, "language": language})
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "audio/wav"))
    parts = list(_chunks(text, speaker))
    if not parts:
        return JSONResponse({"error": "no speech made"}, status_code=503)
    return Response(_wav(np.concatenate(parts)), media_type="audio/wav")


@app.post("/speak_stream")
def speak_stream(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de"),
                 instruct: str = Form(""), mix: str = Form(""), temperature: float = Form(0.7)):
    if _english(language):
        import httpx

        def relay():
            with httpx.stream("POST", EN_URL.rstrip("/") + "/speak_stream",
                              data={"text": text, "speaker": speaker, "language": language}, timeout=120) as r:
                yield from r.iter_lines()
        return StreamingResponse((line + "\n" for line in relay() if line), media_type="application/x-ndjson")

    def lines():
        piece = int(PIECE_S * SR)
        for audio in _chunks(text, speaker):
            for start in range(0, len(audio), piece):
                yield json.dumps({"audio": base64.b64encode(_wav(audio[start:start + piece])).decode()}) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/speakers")
def speakers():
    return {vid: v[0] for vid, v in VOICES.items()}


@app.get("/voices")
def voices():
    return [{"id": vid, "label": f"{v[0]} (apptor-tiny-tts)", "description": f"{v[2]} – Deutsch, 4.5M Parameter, CPU",
             "builtin": True} for vid, v in VOICES.items()]


@app.get("/health")
def health():
    return {"ok": True, "engine": "apptor-tiny-tts", "languages": ["de"] + (["en"] if EN_URL else []),
            "model": MODEL.name, "step": META.get("step"), "vocos": VOCOS or None}


# Warm up once (the first ONNX run allocates its buffers).
_synth("Hallo.", 0)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
