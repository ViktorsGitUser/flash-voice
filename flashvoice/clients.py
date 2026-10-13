"""The services flash-voice talks to: the LLM (OpenAI-compatible), the voice (tts_server.py), Whisper
(stt_server.py) and Clef (llama.cpp's llama-server with Cloudflare's clef-flash, endpoint /v1/systemone)."""
import json
import time

import httpx

from . import config

_http = httpx.Client(timeout=httpx.Timeout(60, connect=5))


# ---------------------------------------------------------------- speech recognition
def transcribe(wav: bytes, fast: bool = False) -> tuple[str, str]:
    r = _http.post(config.STT_URL + "/api/stt", files={"audio": ("speech.wav", wav, "audio/wav")},
                   data={"langs": ",".join(config.LANGS), "fast": "1" if fast else ""})
    r.raise_for_status()
    d = r.json()
    return d.get("text", ""), d.get("lang", config.LANGS[0])


# ---------------------------------------------------------------- Clef, the decision model
_clef = {"t": 0.0, "ok": False}


def clef_ok() -> bool:
    if not config.CLEF_URL:
        return False
    if time.time() - _clef["t"] > 30:
        try:
            ok = _http.get(config.CLEF_URL + "/health", timeout=3).status_code == 200
        except Exception:
            ok = False
        _clef.update(t=time.time(), ok=ok)
    return _clef["ok"]


def decide(state: str, questions: dict) -> dict:
    """Clef: a text + questions with fixed answers -> a probability for every answer, in one pass (~0.3 s)."""
    r = _http.post(config.CLEF_URL + "/v1/systemone", json={"state": state, "questions": questions}, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- the voice
def voices() -> list[dict]:
    try:
        return _http.get(config.TTS_URL + "/voices", timeout=5).json()
    except Exception:
        return []


def speak_wav(text: str, voice: str, lang: str) -> bytes | None:
    try:
        r = _http.post(config.TTS_URL + "/speak", data={"text": text, "speaker": voice, "language": lang}, timeout=60)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


async def speak_pieces(client: httpx.AsyncClient, text: str, voice: str, lang: str):
    """A sentence as base64 WAV pieces of ~0.7 s, as the voice makes them (the first after ~0.4 s)."""
    async with client.stream("POST", config.TTS_URL + "/speak_stream",
                             data={"text": text, "speaker": voice, "language": lang}, timeout=60) as r:
        async for line in r.aiter_lines():
            if line.strip():
                a = json.loads(line).get("audio")
                if a:
                    yield a


# ---------------------------------------------------------------- the brain
_model = {"name": ""}


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.LLM_KEY}"} if config.LLM_KEY else {}


def model_name() -> str:
    if config.LLM_MODEL:
        return config.LLM_MODEL
    if not _model["name"]:
        try:
            data = _http.get(config.LLM_URL + "/models", headers=_headers(), timeout=5).json().get("data") or []
            _model["name"] = data[0]["id"] if data else ""
        except Exception:
            pass
    return _model["name"] or "default"


async def chat_stream(client: httpx.AsyncClient, messages: list):
    """The answer of the LLM as text pieces (OpenAI chat completions, stream). Thinking (reasoning_content, or
    <think>...</think> in the text) is left out - it is not spoken."""
    body = {"model": model_name(), "messages": messages, "stream": True}
    if config.LLM_EXTRA:
        body.update(json.loads(config.LLM_EXTRA))
    async with client.stream("POST", config.LLM_URL + "/chat/completions", json=body, headers=_headers(),
                             timeout=httpx.Timeout(300, connect=10)) as r:
        if r.status_code != 200:
            raise RuntimeError(f"LLM: HTTP {r.status_code} {(await r.aread()).decode('utf-8', 'replace')[:300]}")
        thinking = False
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
            except ValueError:
                continue
            text = delta.get("content") or ""
            while text:   # <think> ... </think> inside the text (some models): skip it
                if thinking:
                    end = text.find("</think>")
                    if end < 0:
                        text = ""
                        break
                    text, thinking = text[end + 8:], False
                else:
                    start = text.find("<think>")
                    if start < 0:
                        break
                    if start:
                        yield text[:start]
                    text, thinking = text[start + 7:], True
            if text:
                yield text
