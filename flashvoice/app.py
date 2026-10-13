"""flash-voice: talk with your LLM - and interrupt it like a person.

    GET  /                  the page (web/index.html)
    GET  /api/info          what is set up (password?, languages, voice, Clef, model)
    POST /api/login         {password} -> session cookie (only when FV_PASSWORD is set)
    GET  /api/voices        the voices of the TTS service
    GET  /api/greeting      ?voice= -> {"audio": base64 WAV} "Hi!" (made once per voice, checked by Whisper)
    WS   /api/ear           the ear (ear.py): microphone in, speech / end / duck / resume / barge out
    POST /api/turn          {messages: [{role, content}], voice, lang, speak} -> NDJSON:
                            {"t": "delta", "text"} {"t": "audio", "text", "audio", "cont"} {"t": "done", "text"}
                            {"t": "error", "text"}
The conversation lives in the browser; the server keeps nothing.
"""
import asyncio
import base64
import datetime
import json
import re
import secrets
import time

import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import clients, config, ear
from .speech import Splitter

app = FastAPI(title="flash-voice")
WEB = config.ROOT / "web"
COOKIE = "fv_session"
_sessions: dict[str, float] = {}
_fails: dict[str, list] = {}
OPEN = ("/api/login", "/api/info")


def log(msg: str):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- password (only when FV_PASSWORD is set)
def session_ok(token: str | None) -> bool:
    if not config.PASSWORD:
        return True
    return bool(token) and _sessions.get(token, 0) > time.time()


@app.middleware("http")
async def gate(request: Request, call_next):
    p = request.url.path
    if p.startswith("/api/") and p not in OPEN and not session_ok(request.cookies.get(COOKIE)):
        return JSONResponse({"error": "login"}, status_code=401)
    return await call_next(request)


@app.post("/api/login")
async def login(request: Request):
    ip = request.client.host if request.client else "?"
    recent = [t for t in _fails.get(ip, []) if time.time() - t < 600]
    if len(recent) >= 5:
        return JSONResponse({"error": "too many attempts - wait 10 minutes"}, status_code=429)
    pw = str((await request.json()).get("password") or "")
    if not config.PASSWORD or not secrets.compare_digest(pw.encode(), config.PASSWORD.encode()):
        _fails[ip] = recent + [time.time()]
        return JSONResponse({"error": "wrong password"}, status_code=403)
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + 30 * 86400
    r = JSONResponse({"ok": True})
    r.set_cookie(COOKIE, token, max_age=30 * 86400, httponly=True, samesite="lax",
                 secure=request.url.scheme == "https")
    return r


@app.get("/api/info")
def info(request: Request):
    return {"auth": bool(config.PASSWORD), "ok": session_ok(request.cookies.get(COOKIE)), "langs": config.LANGS,
            "voice": config.VOICE, "clef": clients.clef_ok(), "model": clients.model_name() if session_ok(
                request.cookies.get(COOKIE)) else ""}


@app.get("/")
def page():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/voices")
def voices():
    return clients.voices()


# ---------------------------------------------------------------- 👋 the greeting
GREETINGS: dict = {}


@app.get("/api/greeting")
def greeting(voice: str = ""):
    """A voice is not the same every time (a take can be empty, or sound like "Bye"): up to 4 takes, the first one
    in which Whisper hears "hi" is kept."""
    voice = voice or config.VOICE
    if not GREETINGS.get(voice):
        for _ in range(4):
            wav = clients.speak_wav("Hi!", voice, config.LANGS[0])
            if not wav:
                break
            GREETINGS[voice] = GREETINGS.get(voice) or wav
            try:
                if re.search(r"\bhi\b", clients.transcribe(wav, fast=True)[0].lower()):
                    GREETINGS[voice] = wav
                    break
            except Exception:
                break
    w = GREETINGS.get(voice)
    return {"audio": base64.b64encode(w).decode() if w else None, "text": "Hi!"}


# ---------------------------------------------------------------- 👂 the ear
@app.websocket("/api/ear")
async def ear_ws(ws: WebSocket):
    if not session_ok(ws.cookies.get(COOKIE)):   # the http middleware does not see WebSockets
        await ws.close(code=4401)
        return
    await ws.accept()
    await ear.serve(ws, clients.transcribe, clients.decide, clients.clef_ok, log, config.LANGS[0])


# ---------------------------------------------------------------- one turn: the LLM's answer, spoken sentence by sentence
SYSTEM = {
    "de": "Du bist ein freundlicher Sprachassistent. Der Mensch spricht mit dir, und deine Antwort wird vorgelesen. "
          "Antworte darum gesprochen und kurz – meist ein bis vier Sätze, mehr nur, wenn er es will. Keine Tabellen, "
          "keine Aufzählungen, kein Markdown, keine Emojis, keine Links. Antworte in der Sprache, in der er spricht. "
          "Heute ist {date}.",
    "en": "You are a friendly voice assistant. The person talks with you, and your answer is read aloud. So answer "
          "the way people speak, and briefly - mostly one to four sentences, more only when asked. No tables, no "
          "lists, no markdown, no emojis, no links. Answer in the language the person speaks. Today is {date}.",
}


def system_prompt(lang: str) -> str:
    base = config.SYSTEM_PROMPT or SYSTEM.get(lang, SYSTEM["en"])
    return base.replace("{date}", datetime.date.today().strftime("%d.%m.%Y" if lang == "de" else "%Y-%m-%d"))


@app.post("/api/turn")
async def turn(request: Request):
    j = await request.json()
    lang = str(j.get("lang") or config.LANGS[0])[:5]
    voice = str(j.get("voice") or config.VOICE)
    speak = bool(j.get("speak", True))
    msgs = [{"role": m["role"], "content": str(m["content"])[:20000]} for m in (j.get("messages") or [])[-40:]
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
    if not msgs or msgs[-1]["role"] != "user":
        return JSONResponse({"error": "the last message must be the person's"}, status_code=400)
    msgs = [{"role": "system", "content": system_prompt(lang)}] + msgs

    async def lines():
        q: asyncio.Queue = asyncio.Queue()
        sentences: asyncio.Queue = asyncio.Queue()
        async with httpx.AsyncClient() as client:
            async def brain():
                full, sp = [], Splitter()
                try:
                    async for d in clients.chat_stream(client, msgs):
                        full.append(d)
                        await q.put({"t": "delta", "text": d})
                        for p in sp.feed(d) if speak else []:
                            await sentences.put(p)
                    for p in sp.flush() if speak else []:
                        await sentences.put(p)
                    await q.put({"t": "done", "text": "".join(full)})
                except Exception as e:
                    log(f"[turn] LLM: {e}")
                    await q.put({"t": "error", "text": str(e)[:300]})
                finally:
                    await sentences.put(None)

            async def voice_task():   # one sentence after the other, each as soon as it is complete
                while (s := await sentences.get()) is not None:
                    first = True
                    try:
                        async for a in clients.speak_pieces(client, s, voice, lang):
                            await q.put({"t": "audio", "text": s if first else "", "audio": a, "cont": not first})
                            first = False
                    except Exception as e:
                        log(f"[turn] voice: {e}")
                await q.put(None)

            tasks = [asyncio.create_task(brain()), asyncio.create_task(voice_task())]
            try:
                while (item := await q.get()) is not None:
                    yield json.dumps(item, ensure_ascii=False) + "\n"
            finally:   # the page went away or interrupted: stop the LLM and the voice
                for t in tasks:
                    t.cancel()

    return StreamingResponse(lines(), media_type="application/x-ndjson")
