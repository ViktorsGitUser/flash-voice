"""flash-voice: apptor-tiny-tts, a tiny German CPU voice built on tiny-voca - the same HTTP API as tts_server.py and phonon_server.py.

tiny-voca (github.com/n4ze3m/tiny-voca) is a 4.5M-parameter TTS with its own waveform generator: ~18 MB, no GPU,
many times faster than real time on a few CPU threads. apptor-tiny-tts is a German model (not an official tiny-voca release) trained with that recipe on
synthetic speech from Qwen3-TTS (two voices, F2 female and M2 male, 24 kHz). Text -> phonemes with the own
letter-to-sound model apptor-g2p-de (Apache 2.0, trained to reproduce eSpeak NG "de"), or eSpeak NG itself.

License: code and German weights Apache 2.0 (unlike phonon-40m-de, which is CC BY-NC 4.0 - non-commercial only).
With models/apptor-g2p-de.onnx no espeak-ng program is needed; without it the server falls back to espeak-ng
(GPL-3.0, called as a separate process, not bundled).

apptor-tiny-tts-at (models/apptor-tiny-tts-at.onnx) adds an Austrian accent switch: voices f2/m2 with -de, -at,
-wien, -oarg (accent 0, 1.0, 1.3, 1.6), plain f2/m2 use FV_TINYTTS_ACCENT; the form field "accent" overrides.

The model is German only. An English request is passed on to FV_TINYTTS_EN_URL (e.g. the Phonon server, whose
English model is Apache 2.0) if that is set; otherwise the German voice reads it.

POST /speak         form: text, speaker, language (de|en) -> audio/wav
POST /speak_stream  form: text, speaker, language -> NDJSON lines {"audio": base64 WAV} of ~0.5 s each
GET  /voices, /speakers, /health

Second stage (FV_TINYTTS_VOCOS): the waveform of the tiny-voca model is re-synthesised by Vocos (MIT, 13.5M
parameters, CPU) from its mel spectrogram - this removes most of the buzzy sound of the tiny source-filter generator.
Default: models/apptor-vocos-de.int8.onnx (fine-tuned on the two voices, ONNX int8, no torch) when it is there.
FV_TINYTTS_VOCOS=models/apptor-tiny-tts-vocos-de.pt uses the PyTorch version; =pretrained the generic
charactr/vocos-mel-24khz; both need torch and the vocos package (requirements-tinytts-vocos.txt). "" switches it off.

Run:  python services/tinytts_server.py   (FV_TINYTTS_PORT 8089, FV_TINYTTS_MODEL models/apptor-tiny-tts-de.onnx,
      FV_TINYTTS_THREADS 4, FV_TINYTTS_SPEED 1.0, FV_TINYTTS_VOCOS models/apptor-vocos-de.int8.onnx)
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
ROOT = Path(__file__).resolve().parent.parent
# Letters -> phonemes: the own model (apptor-g2p-de.onnx, Apache 2.0, no eSpeak NG needed) when it is there,
# else the espeak-ng program as before. FV_TINYTTS_G2P=espeak forces the old way.
G2P = os.environ.get("FV_TINYTTS_G2P", str(ROOT / "models" / "apptor-g2p-de.onnx"))
if G2P != "espeak" and Path(G2P).is_file():
    os.environ["APPTOR_G2P_MODEL"] = G2P
    from tinyvoca.g2p_de import phonemize  # noqa: E402
    FRONTEND = "apptor-g2p-de"
else:
    from tinyvoca.text import phonemize  # noqa: E402  (vendored tiny-voca front end, German, eSpeak NG)
    FRONTEND = "espeak-ng"

PORT = int(os.environ.get("FV_TINYTTS_PORT", "8089"))
HOST = os.environ.get("FV_TINYTTS_HOST", "127.0.0.1")
MODEL = Path(os.environ.get("FV_TINYTTS_MODEL", ROOT / "models" / "apptor-tiny-tts-de.onnx"))
THREADS = int(os.environ.get("FV_TINYTTS_THREADS", "4"))   # Strata needs the CPU too
SPEED = float(os.environ.get("FV_TINYTTS_SPEED", "1.0"))
EN_URL = os.environ.get("FV_TINYTTS_EN_URL", "")         # e.g. http://127.0.0.1:8085 (Phonon, English)
# second stage: an ONNX file (no torch needed; default when models/apptor-vocos-de.int8.onnx is there), "", "pretrained"
# or a fine-tuned PyTorch state dict (.pt)
_VOCOS_ONNX = ROOT / "models" / "apptor-vocos-de.int8.onnx"
VOCOS = os.environ.get("FV_TINYTTS_VOCOS", str(_VOCOS_ONNX) if _VOCOS_ONNX.is_file() else "")
PIECE_S = 0.5
GAP_S = 0.12   # silence between the parts of a long sentence

META = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))
SR = META["sample_rate"]
MAX_TOKENS = META["max_tokens"]

# id -> (label, model voice, description, accent). The Phonon ids are kept, so a client that chose a Phonon voice
# still gets one of the same kind. A model with the accent switch (apptor-tiny-tts-at: input "accent") adds the
# Austrian variants; plain f2/m2 then speak with FV_TINYTTS_ACCENT.
ACCENT_MODEL = "accent" in META.get("inputs", [])
ACCENT = float(os.environ.get("FV_TINYTTS_ACCENT", "1.3" if ACCENT_MODEL else "0"))
ACCENT_MAX = 1.6   # above that the pronunciation breaks (2.0 sounds broken)
VOICES = {
    "f2": ("Tiny F2" + (f" (Standard, Akzent {ACCENT:g})" if ACCENT_MODEL else ""), "F2", "weiblich, klar", ACCENT),
    "m2": ("Tiny M2" + (f" (Standard, Akzent {ACCENT:g})" if ACCENT_MODEL else ""), "M2", "männlich, ruhig", ACCENT),
}
if ACCENT_MODEL:
    for vid, voice, kind in (("f2", "F2", "weiblich"), ("m2", "M2", "männlich")):
        for suffix, label, desc, accent in (("de", "Hochdeutsch", "hochdeutsch", 0.0),
                                            ("at", "Österreich", "österreichisch", 1.0),
                                            ("wien", "Wien", "österreichisch mit Wiener Note", 1.3),
                                            ("oarg", "Oarg", "kräftig österreichisch", 1.6)):
            VOICES[f"{vid}-{suffix}"] = (f"Tiny {voice} {label}", voice, f"{kind}, {desc}", accent)
ALIASES = {"annika": "f2", "lotta": "f2", "nele": "f2", "henrik": "m2", "leon": "m2", "mika": "m2"}
DEFAULT = os.environ.get("FV_TINYTTS_DEFAULT", "f2")

options = ort.SessionOptions()
options.intra_op_num_threads = THREADS
options.inter_op_num_threads = 1
print(f"[tiny-tts] loading {MODEL} ({THREADS} threads) ...", flush=True)
_session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
_vocos = None
_vocos_onnx = None


def _vocos_kernels(n_fft=1024):
    """Fixed STFT / inverse STFT kernels of the Vocos ONNX model (periodic Hann window)."""
    n = np.arange(n_fft)
    k = np.arange(n_fft // 2 + 1)[:, None]
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)
    ang = 2 * np.pi * k * n / n_fft
    c = np.where((k == 0) | (k == n_fft // 2), 1.0, 2.0) / n_fft
    return {"a_real": (w * np.cos(ang))[:, None].astype(np.float32),
            "a_imag": (-w * np.sin(ang))[:, None].astype(np.float32),
            "synth": np.concatenate((c * np.cos(ang) * w, -c * np.sin(ang) * w))[:, None].astype(np.float32)}


if VOCOS.endswith(".onnx"):
    _vocos_onnx = ort.InferenceSession(str(Path(VOCOS) if Path(VOCOS).is_absolute() else ROOT / VOCOS), options,
                                       providers=["CPUExecutionProvider"])
    _kernels = _vocos_kernels()
    print(f"[tiny-tts] second stage: Vocos ONNX ({VOCOS})", flush=True)
elif VOCOS:
    import torch
    from vocos import Vocos
    torch.set_num_threads(THREADS)
    _vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
    if VOCOS != "pretrained":
        path = Path(VOCOS) if Path(VOCOS).is_absolute() else ROOT / VOCOS
        _vocos.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    print(f"[tiny-tts] second stage: Vocos ({VOCOS})", flush=True)
print(f"[tiny-tts] letters -> phonemes: {FRONTEND}", flush=True)
app = FastAPI(title="flash-voice apptor-tiny-tts")


def _voice(speaker: str, accent: str = "") -> tuple[int, float]:
    """Speaker id -> (model voice index, accent strength). ``accent`` (form field, e.g. from a slider) overrides."""
    vid = (speaker or "").lower()
    vid = ALIASES.get(vid, vid)
    _, voice, _, strength = VOICES.get(vid, VOICES[DEFAULT])
    if accent.strip():
        try:
            strength = float(accent)
        except ValueError:
            pass
    return META["voices"].index(voice), min(max(strength, 0.0), ACCENT_MAX) if ACCENT_MODEL else 0.0


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


def _phonemize(text: str, accent: float = 0.0):
    """Austrian readings (Kaffee, Chemie ...) from accent 0.5 on - only the own front end has them."""
    if FRONTEND == "apptor-g2p-de":
        return phonemize(text, austrian=accent >= 0.5)
    return phonemize(text)


def _trim_tail(audio: np.ndarray, floor_db: float = -30.0, fade_ms: float = 40.0) -> np.ndarray:
    """Cut the breathy noise after the last sound (the Austrian style learned it from Common Voice; after a final "p"
    it sounds like an "f"): keep up to the last 10 ms frame within 30 dB of the loudest, then fade out 40 ms."""
    hop = SR // 100
    n = len(audio) // hop
    if n == 0:
        return audio
    e = (audio[:n * hop].reshape(n, hop) ** 2).sum(1)
    loud = np.nonzero(e >= e.max() * 10 ** (floor_db / 10))[0]
    fade = int(fade_ms * SR / 1000)
    end = (int(loud[-1]) + 1) * hop + fade
    if e.max() <= 0 or end >= len(audio):
        return audio
    out = audio[:end].copy()
    out[-fade:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(1, fade + 1) / fade)
    return out


def _synth(text: str, voice: int, speed: float = SPEED, accent: float = 0.0) -> np.ndarray:
    phones = _phonemize(text, accent)
    feeds = {
        "tokens": np.array([phones.ids], dtype=np.int64),
        "stress": np.array([phones.stress], dtype=np.int64),
        "voice": np.array([voice], dtype=np.int64),
        "speed": np.array([speed], dtype=np.float32),
        "pitch_shift": np.array([0.0], dtype=np.float32),
        "pitch_range": np.array([1.0], dtype=np.float32),
    }
    if ACCENT_MODEL:
        feeds["accent"] = np.array([accent], dtype=np.float32)
    audio, _ = _session.run(None, feeds)
    audio = audio[0].astype(np.float32)
    if _vocos_onnx is not None:
        audio = _vocos_onnx.run(None, {"audio": audio[None], **_kernels})[0][0, :len(audio)]
    elif _vocos is not None:
        with torch.inference_mode():
            x = torch.from_numpy(audio)[None]
            audio = _vocos.decode(_vocos.feature_extractor(x))[0, :len(audio)].numpy()
    audio = _trim_tail(audio)
    peak = float(np.abs(audio).max())
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return audio.astype(np.float32)


def _chunks(text: str, speaker: str, accent: str = ""):
    voice, strength = _voice(speaker, accent)
    for i, part in enumerate(_parts(text)):
        if i:
            yield np.zeros(int(GAP_S * SR), dtype=np.float32)
        yield _synth(part, voice, accent=strength)


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
          mix: str = Form(""), temperature: float = Form(0.7), accent: str = Form("")):
    if _english(language):
        r = _forward("/speak", {"text": text, "speaker": speaker, "language": language})
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "audio/wav"))
    parts = list(_chunks(text, speaker, accent))
    if not parts:
        return JSONResponse({"error": "no speech made"}, status_code=503)
    return Response(_wav(np.concatenate(parts)), media_type="audio/wav")


@app.post("/speak_stream")
def speak_stream(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de"),
                 instruct: str = Form(""), mix: str = Form(""), temperature: float = Form(0.7), accent: str = Form("")):
    if _english(language):
        import httpx

        def relay():
            with httpx.stream("POST", EN_URL.rstrip("/") + "/speak_stream",
                              data={"text": text, "speaker": speaker, "language": language}, timeout=120) as r:
                yield from r.iter_lines()
        return StreamingResponse((line + "\n" for line in relay() if line), media_type="application/x-ndjson")

    def lines():
        piece = int(PIECE_S * SR)
        for audio in _chunks(text, speaker, accent):
            for start in range(0, len(audio), piece):
                yield json.dumps({"audio": base64.b64encode(_wav(audio[start:start + piece])).decode()}) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/speakers")
def speakers():
    return {vid: v[0] for vid, v in VOICES.items()}


@app.get("/voices")
def voices():
    return [{"id": vid, "label": f"{v[0]} (apptor-tiny-tts)", "description": f"{v[2]} – Deutsch, 4.5M Parameter, CPU",
             "builtin": True, **({"accent": v[3]} if ACCENT_MODEL else {})} for vid, v in VOICES.items()]


@app.get("/health")
def health():
    return {"ok": True, "engine": "apptor-tiny-tts", "languages": ["de"] + (["en"] if EN_URL else []),
            "model": MODEL.name, "step": META.get("step"), "vocos": VOCOS or None, "frontend": FRONTEND,
            "accent": {"default": ACCENT, "max": ACCENT_MAX} if ACCENT_MODEL else None}


# Warm up once (the first ONNX run allocates its buffers).
_synth("Hallo.", 0, accent=ACCENT)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
