"""OpenAI-compatible audio API in front of flash-voice's voice and ears - for Open WebUI (Settings -> Audio ->
engine "OpenAI") and any other app that speaks the OpenAI audio API. Works with every flash-voice voice
(apptor-tiny-tts, Phonon, Qwen3-TTS) and both ears (Fluid Speech Pro, Whisper).

POST /v1/audio/speech          JSON {input, voice, model, response_format}  -> audio (mp3 when ffmpeg is there, or wav)
POST /v1/audio/transcriptions  form: file, model, language, response_format -> {"text"}
GET  /v1/audio/voices, /v1/audio/models, /v1/models, /health

The OpenAI API sends no language with the text: it is guessed from the words (among FV_OAI_LANGS), so an English
sentence goes to the English voice and a German one to the German voice.

Run:  python services/openai_audio.py   (FV_OAI_PORT 8090, FV_OAI_HOST 127.0.0.1, FV_TTS_URL :8086, FV_STT_URL :8087,
      FV_OAI_LANGS en,de, FV_OAI_KEY: if set, requests need "Authorization: Bearer <key>", FV_OAI_VOICE: default voice)
"""
import os
import re
import shutil
import subprocess

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

PORT = int(os.environ.get("FV_OAI_PORT", "8090"))
HOST = os.environ.get("FV_OAI_HOST", "127.0.0.1")
TTS = os.environ.get("FV_TTS_URL", "http://127.0.0.1:8086").rstrip("/")
STT = os.environ.get("FV_STT_URL", "http://127.0.0.1:8087").rstrip("/")
LANGS = tuple(l.strip() for l in os.environ.get("FV_OAI_LANGS", "en,de").split(",") if l.strip())
KEY = os.environ.get("FV_OAI_KEY", "")
VOICE = os.environ.get("FV_OAI_VOICE", "")
FFMPEG = shutil.which("ffmpeg")
MEDIA = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac"}

app = FastAPI(title="flash-voice OpenAI audio")
_http = httpx.Client(timeout=120)

COMMON = {
    "en": set("the and is are not you he she it we they a an to with on for of in yes no also what how where can please "
              "thanks still already but or if that have has was me my your hello today i this do does there would "
              "could should will about from".split()),
    "de": set("der die das und ist nicht ich du er sie es wir ihr ein eine zu mit auf für von den dem im ja nein auch "
              "was wie wo kann bitte danke noch schon aber oder wenn dass hab habe sind war mir mich dir dich hallo "
              "heute sich bei nach auch wird werden einen".split()),
    "hr": set("je i u na da se ne to su sam si smo što kako gdje može molim hvala još već ali ili ako".split()),
}


def guess_lang(text: str) -> str:
    words = re.findall(r"\w+", text.lower())
    score = {l: sum(w in COMMON.get(l, ()) for w in words) for l in LANGS}
    if "de" in score:
        score["de"] += 2 * sum(c in "äöüß" for c in text.lower())
    if "hr" in score:
        score["hr"] += 2 * sum(c in "čćšžđ" for c in text.lower())
    return max(LANGS, key=lambda l: (score.get(l, 0), -LANGS.index(l)))


def check(auth: str | None):
    if KEY and auth != f"Bearer {KEY}":
        raise HTTPException(401, "wrong or missing API key")


def encode(wav: bytes, fmt: str) -> tuple[bytes, str]:
    if fmt in ("wav", "pcm") or not FFMPEG or fmt not in MEDIA:
        return wav, "audio/wav"
    codec = {"mp3": ["-f", "mp3", "-b:a", "64k"], "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "32k"],
             "aac": ["-f", "adts"], "flac": ["-f", "flac"]}[fmt]
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-i", "pipe:0", *codec, "pipe:1"], input=wav, capture_output=True)
    return (r.stdout, MEDIA[fmt]) if r.returncode == 0 and r.stdout else (wav, "audio/wav")


def tts_voices() -> list[dict]:
    try:
        return _http.get(TTS + "/voices", timeout=5).json()
    except Exception:
        return []


@app.post("/v1/audio/speech")
async def speech(request: Request, authorization: str | None = Header(None)):
    check(authorization)
    body = await request.json()
    text = str(body.get("input", "")).strip()
    if not text:
        raise HTTPException(400, "input is empty")
    voice = str(body.get("voice") or VOICE or "")
    if voice.lower() in ("alloy", "echo", "fable", "onyx", "nova", "shimmer"):   # OpenAI's names: use the default
        voice = VOICE
    lang = guess_lang(text)
    r = _http.post(TTS + "/speak", data={"text": text, "speaker": voice, "language": lang})
    if r.status_code != 200:
        return JSONResponse({"error": f"voice service: {r.status_code}"}, status_code=502)
    audio, media = encode(r.content, str(body.get("response_format") or "mp3"))
    return Response(audio, media_type=media)


@app.post("/v1/audio/transcriptions")
def transcriptions(file: UploadFile = File(...), model: str = Form(""), language: str = Form(""),
                   response_format: str = Form("json"), authorization: str | None = Header(None)):
    check(authorization)
    langs = language.strip().lower() or ",".join(LANGS)
    r = _http.post(STT + "/api/stt", files={"audio": (file.filename or "audio.webm", file.file.read())},
                   data={"langs": langs})
    if r.status_code != 200:
        return JSONResponse({"error": f"speech recognition: {r.status_code}"}, status_code=502)
    d = r.json()
    if response_format == "text":
        return PlainTextResponse(d.get("text", ""))
    out = {"text": d.get("text", "")}
    if response_format == "verbose_json":
        out["language"] = d.get("lang", "")
    return out


@app.get("/v1/audio/voices")
def voices(authorization: str | None = Header(None)):
    check(authorization)
    return {"voices": [{"id": v["id"], "name": v.get("label") or v["id"]} for v in tts_voices() if v.get("id")]}


@app.get("/v1/audio/models")
def audio_models():
    return {"models": [{"id": "tts-1"}, {"id": "whisper-1"}]}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "flash-voice"}
                                       for m in ("tts-1", "whisper-1")]}


@app.get("/health")
def health():
    def ok(url):
        try:
            return _http.get(url + "/health", timeout=3).status_code == 200
        except Exception:
            return False
    return {"ok": True, "voice": ok(TTS), "ears": ok(STT), "mp3": bool(FFMPEG), "langs": LANGS}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
