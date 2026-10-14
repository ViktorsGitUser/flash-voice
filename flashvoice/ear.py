"""The ear: listening the whole time - also while the assistant speaks.

The page streams the microphone over a WebSocket (16 kHz, 16-bit mono) and says what it is doing (mode: listening /
thinking / speaking) and what the assistant is saying right now (for the echo). Here, frame by frame (32 ms):

* Silero VAD (MIT, onnxruntime on the CPU, ~0.1 ms per frame): a human voice, or noise?
* LISTENING: the utterance starts at the first voice (with 0.4 s before it, so the first syllable is kept). At the
  first pause (0.4 s) Whisper writes it down and Clef is asked "is the thought finished?" - yes: the answer starts at
  once; no ("I'd like, uhm ..."): wait up to 1.8 s of silence. Without Clef: 1.1 s of silence.
* SPEAKING / THINKING: a voice for 160 ms -> "duck" (the page turns the assistant down to 25 %). Whisper writes down
  the first second: the assistant's own sentence coming back through the microphone (echo), or nothing -> "resume".
  "mhm", "yes", "right" ... -> "resume". Otherwise Clef: interruption / just listening / talking to someone else?
  Interruption -> "barge": the page stops the assistant, and the utterance goes on being heard from its first syllable.

Messages to the page: ready, speech, reset, end {text, lang}, duck, resume, barge. Nothing is stored; the log only
has the steps, never what was said.
"""
import asyncio
import collections
import io
import json
import re
import time
import wave
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 512                      # 32 ms at 16 kHz (Silero v5)
FRAME_MS = FRAME * 1000 / SR
MODEL = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"
PRE_ROLL = 13                    # frames kept before the voice starts (~0.4 s)
START_P, KEEP_P = 0.5, 0.35      # a voice starts above START_P and goes on above KEEP_P
LISTEN_START_FRAMES = 3          # ~100 ms of voice: someone speaks
BARGE_START_FRAMES = 5           # while the assistant speaks: ~160 ms (echo remains and clicks are shorter)
PAUSE_CHECK_MS = 400             # first pause: write it down, ask Clef whether the thought is finished
END_SURE_MS = 1100               # end of turn without Clef
END_MAX_MS = 1800                # Clef said "not finished": wait this long at most
MAX_UTTER_S = 30
BARGE_DECIDE_MS = 700            # voice this long while the assistant speaks: decide (or earlier, when it stops)
BARGE_FORCE_MS = 2500            # still speaking after this long: an interruption, whatever else
BACKCHANNEL = re.compile(
    r"(?i)^\W*(m+h*m+|hm+|uh[- ]?huh|aha|ah|oh|ja+|jo|jaja|genau|okay|ok|gut|stimmt|richtig|super|alles klar|verstehe|"
    r"achso|ach so|na ja|nein|nee|danke|yes|yeah|yep|yup|right|sure|cool|nice|i see|got it|true|no|nope|thanks)\W*$")
# Whisper's well-known hallucinations on silence and noise
JUNK = re.compile(r"(?i)\W*(untertitel.*|vielen dank\.?|thank you\.?|thanks for watching.*|bye\.?|tschüss\.?|"
                  r"subtitles by.*|amara\.org.*)\W*")
TEXTS = {
    "de": {
        "said": "Der Mensch sagt: „{t}“",
        "finished": "Ist die Äußerung des Menschen ein abgeschlossener Gedanke (fertige Frage oder Aussage), auf den man "
                    "jetzt antworten kann? Nein, wenn der Satz mitten drin aufhört oder er offensichtlich noch weiterreden will.",
        "barge_state": "Der Sprachassistent sagt gerade: „{said}“\nDer Mensch sagt dazwischen: „{t}“",
        "barge_q": "Was tut der Mensch, während der Assistent spricht?",
        "interrupts": "Er spricht den Assistenten an: eine Frage, ein Einwand, eine Korrektur, ein Wunsch, oder er sagt stopp/warte",
        "listens": "Nur ein kurzes Zuhör-Signal wie mhm, ja, genau, okay, aha, oder ein Lachen",
        "other": "Er redet mit einer anderen Person oder es ist Hintergrund (Radio, Fernseher)",
    },
    "en": {
        "said": "The person says: \"{t}\"",
        "finished": "Is what the person said a finished thought (a complete question or statement) that can be answered "
                    "now? No if the sentence stops in the middle or they obviously want to go on.",
        "barge_state": "The voice assistant is saying: \"{said}\"\nThe person says in between: \"{t}\"",
        "barge_q": "What is the person doing while the assistant speaks?",
        "interrupts": "Addressing the assistant: a question, an objection, a correction, a wish, or saying stop/wait",
        "listens": "Only a short listening signal like mhm, yes, right, okay, or a laugh",
        "other": "Talking to another person, or it is background (radio, TV)",
    },
}


def _texts(lang: str) -> dict:
    return TEXTS.get(lang, TEXTS["en"])


def finished_q(lang: str) -> dict:
    return {"f": {"type": "noul", "instructions": _texts(lang)["finished"]}}


def barge_q(lang: str) -> dict:
    t = _texts(lang)
    return {"art": {"type": "choice", "instructions": t["barge_q"], "criteria": {
        "interrupts": t["interrupts"], "listens": t["listens"], "other": t["other"]}}}


_session = None


def _vad_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1   # one CPU thread is plenty (~0.1 ms per frame)
        o.inter_op_num_threads = 1
        _session = ort.InferenceSession(str(MODEL), sess_options=o, providers=["CPUExecutionProvider"])
    return _session


class Vad:
    """Silero VAD v5 on 512-sample frames (it wants the last 64 samples of the frame before as context)."""
    def __init__(self):
        self.s = _vad_session()
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros((1, 64), dtype=np.float32)
        self.sr = np.array(SR, dtype=np.int64)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.ctx, frame.reshape(1, -1)], axis=1)
        out, self.state = self.s.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.ctx = x[:, -64:]
        return float(out[0][0])


def wav_bytes(frames: list) -> bytes:
    pcm = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _words(t: str) -> list[str]:
    return re.findall(r"\w+", (t or "").lower())


def is_echo(heard: str, said: str) -> bool:
    """What the microphone heard is (mostly) the assistant's own voice: most of its words are in what it just said."""
    h, s = _words(heard), set(_words(said))
    if not h or not s:
        return False
    return sum(w in s for w in h) / len(h) >= 0.6


class Ear:
    """One per WebSocket. transcribe(wav, fast) -> (text, lang); decide(state, questions) -> Clef's answers;
    clef_ok() -> bool; log(str)."""
    def __init__(self, ws, transcribe, decide, clef_ok, log, lang: str = "de"):
        self.ws, self.transcribe, self.decide, self.clef_ok, self.log = ws, transcribe, decide, clef_ok, log
        self.lang = lang                       # the language last heard (Clef's questions in the same language)
        self.vad = Vad()
        self.mode = "listening"
        self.pre = collections.deque(maxlen=PRE_ROLL)
        self.buf = np.zeros(0, dtype=np.float32)
        self.said = collections.deque(maxlen=4)     # the assistant's last sentences (echo)
        self.barged = 0.0
        self.reset()

    def reset(self):
        self.gen = getattr(self, "gen", 0) + 1   # checks still running for an earlier utterance see that they are old
        self.utt = None          # frames of the utterance being heard
        self.run = 0             # frames of voice in a row (before it counts as speech)
        self.voiced = 0.0        # ms of voice in the utterance
        self.silence = 0.0       # ms of silence since the last voice
        self.checked = None      # (frames, text, lang, finished) of the last pause check
        self.task = None         # a running Whisper/Clef check
        self.barge = None        # while the assistant speaks: None, "ducked", "deciding"
        self.last_voice = 0      # index of the last frame with voice in utt
        self.ptask = None        # a running live transcript (partial)
        self.plast = 0.0         # when the last one started

    async def send(self, **k):
        try:
            await self.ws.send_text(json.dumps(k, ensure_ascii=False))
        except Exception:
            pass

    # ---------------------------------------------------------------- what the page says
    async def control(self, m: dict):
        if m.get("t") == "mode":
            new = m.get("mode") or "listening"
            if new == self.mode:
                return
            if new in ("speaking", "thinking") and self.utt is not None and time.time() - self.barged < 2:
                return   # a late "speaking" right after a barge: the person is being heard
            old, self.mode = self.mode, new
            if new == "listening" and old in ("speaking", "thinking") and self.barge:
                return   # the page follows a barge: the utterance being heard goes on
            if {old, new} == {"speaking", "thinking"}:
                return   # both are the assistant's turn: a voice being checked stays in its check
            if self.barge in ("ducked", "deciding"):
                self.log(f"mode {old} -> {new}: check dropped")
                await self.send(e="resume")
            self.reset()
        elif m.get("t") == "said":
            self.said.append(str(m.get("text") or "")[:500])

    # ---------------------------------------------------------------- the microphone
    async def audio(self, data: bytes):
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
        self.buf = np.concatenate([self.buf, x])
        while len(self.buf) >= FRAME:
            f, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            await self.frame(f)

    async def frame(self, f: np.ndarray):
        p = self.vad(f)
        if self.mode == "off":   # interrupting switched off: deaf while the assistant has its turn
            return
        if self.utt is None:
            self.pre.append(f)
            need = LISTEN_START_FRAMES if self.mode == "listening" else BARGE_START_FRAMES
            self.run = self.run + 1 if p >= START_P else 0
            if self.run >= need:
                self.utt, self.voiced, self.silence = list(self.pre), self.run * FRAME_MS, 0.0
                self.pre.clear()
                if self.mode == "listening":
                    await self.send(e="speech")
                else:   # the assistant speaks or thinks: turn it down, then find out what it is
                    self.barge = "ducked"
                    await self.send(e="duck")
                    self.log("voice while the assistant speaks - quieter")
            return
        self.utt.append(f)
        if p >= KEEP_P:
            self.voiced += FRAME_MS
            self.silence = 0.0
            self.last_voice = len(self.utt)
            if self.checked and self.checked[0] < self.last_voice:
                self.checked = None   # spoke on after the pause: the old check is out of date
        else:
            self.silence += FRAME_MS
        if self.mode == "listening":
            if (self.ptask is None and self.task is None and self.voiced >= 600 and self.silence < 300
                    and time.time() - self.plast >= 1.0):
                self.plast = time.time()
                self.ptask = asyncio.create_task(self.partial(list(self.utt), self.gen))
            await self.listening()
        else:
            await self.speaking()

    # ---------------------------------------------------------------- the person's turn
    async def listening(self):
        if len(self.utt) * FRAME_MS > MAX_UTTER_S * 1000:
            return await self.finish()
        if self.voiced < 200 and self.silence >= 600:   # a click, a cough: not speech
            self.log("too short - dropped")
            self.reset()
            return await self.send(e="reset")
        if self.silence >= PAUSE_CHECK_MS and self.checked is None and self.task is None:
            self.task = asyncio.create_task(self.pause_check(len(self.utt), self.gen))
        if self.checked and self.checked[3] is True and self.silence >= PAUSE_CHECK_MS:
            return await self.finish()
        if self.checked:   # Clef: not finished -> wait longer; no answer from Clef -> the usual pause
            limit = END_MAX_MS if self.checked[3] is False else END_SURE_MS
        else:              # the check is still running (or there is no Clef)
            limit = END_MAX_MS if self.clef_ok() else END_SURE_MS
        if self.silence >= limit and self.task is None:
            await self.finish()

    async def partial(self, frames: list, gen: int):
        """What has been said so far (Whisper's quick mode, the last 20 s) - the page shows it while the person speaks."""
        try:
            text, _ = await asyncio.to_thread(lambda: self.transcribe(wav_bytes(frames[-625:]), True))
            if gen == self.gen and self.utt is not None and self.mode == "listening" and text.strip():
                await self.send(e="partial", text=text.strip())
        except Exception:
            pass
        finally:
            if gen == self.gen:
                self.ptask = None

    async def pause_check(self, n: int, gen: int):
        """At the first pause: write it down, ask Clef whether the thought is finished."""
        try:
            t0 = time.time()
            text, lang = await asyncio.to_thread(self.transcribe, wav_bytes(self.utt[:n]), False)
            t1 = time.time()
            finished = None
            if self.clef_ok() and text.strip() and not JUNK.fullmatch(text):
                try:
                    r = await asyncio.to_thread(self.decide, _texts(lang)["said"].format(t=text), finished_q(lang))
                    finished = r["answers"]["f"]["noul"] >= 0.6
                except Exception:
                    finished = None
            if gen != self.gen:
                return
            if self.utt is not None and self.last_voice <= n:
                self.checked = (n, text, lang, finished)
                self.log(f"pause: {'finished' if finished else 'not finished' if finished is False else 'no Clef'} "
                         f"(Whisper {t1 - t0:.2f} s, Clef {time.time() - t1:.2f} s)")
        except Exception as e:
            self.log(f"pause check: {e}")
            if gen == self.gen:   # no second try at every frame: the end of the turn comes by the pause alone
                self.checked = (n, None, self.lang if hasattr(self, "lang") else "de", None)
        finally:
            if gen == self.gen:
                self.task = None
        if gen == self.gen and self.utt is not None and self.mode == "listening":
            await self.listening()

    async def finish(self):
        frames, chk, last = self.utt, self.checked, self.last_voice
        self.reset()
        self.mode = "thinking"   # until the page says otherwise
        if chk and chk[1] is not None and chk[0] >= last:   # nothing said since the pause check: its text
            text, lang = chk[1], chk[2]
        else:
            try:
                text, lang = await asyncio.to_thread(self.transcribe, wav_bytes(frames), False)
            except Exception as e:
                self.log(f"speech recognition: {e}")
                text, lang = "", self.lang
        if not text.strip() or JUNK.fullmatch(text):
            self.log("nothing understandable - dropped")
            self.mode = "listening"
            return await self.send(e="reset")
        self.lang = lang or self.lang
        self.log(f"utterance finished ({len(frames) * FRAME_MS / 1000:.1f} s)")
        await self.send(e="end", text=text.strip(), lang=lang)

    # ---------------------------------------------------------------- while the assistant speaks: interruption?
    async def speaking(self):
        if self.barge == "ducked":
            if self.voiced >= BARGE_DECIDE_MS or (self.silence >= 350 and self.voiced >= 100):
                self.barge = "deciding"
                self.task = asyncio.create_task(self.barge_check(len(self.utt), self.gen))
            elif self.silence >= 600:   # too short to be anything
                self.log("too short - go on")
                await self.send(e="resume")
                self.reset()
        elif self.barge == "deciding" and self.voiced >= BARGE_FORCE_MS:
            await self.do_barge("speaks long")

    async def barge_check(self, n: int, gen: int):
        try:
            text, lang = await asyncio.to_thread(self.transcribe, wav_bytes(self.utt[:n]), True)
            lang = lang or self.lang
            said = " ".join(self.said)
            if gen != self.gen or self.barge != "deciding":
                return
            if not text.strip() or JUNK.fullmatch(text) or is_echo(text, said):
                self.log("echo or nothing - go on")
                await self.send(e="resume")
                return self.reset()
            words = _words(text)
            if BACKCHANNEL.fullmatch(text.strip()):
                verdict = "listens"
            elif len(words) >= 4 and self.voiced >= 1000:
                verdict = "interrupts"
            elif self.clef_ok():
                try:
                    r = await asyncio.to_thread(self.decide, _texts(lang)["barge_state"].format(said=said[-400:], t=text),
                                                barge_q(lang))
                    verdict = r["answers"]["art"]["choice"]
                except Exception:
                    verdict = "interrupts"
            else:
                verdict = "interrupts" if len(words) >= 2 else "listens"
            if gen != self.gen or self.barge != "deciding":
                return
            if verdict == "interrupts":
                await self.do_barge("Clef: interruption")
            else:
                self.log(f"{verdict} - go on")
                await self.send(e="resume")
                self.reset()
        except Exception as e:
            self.log(f"interruption check: {e}")
            if gen == self.gen and self.barge:
                await self.send(e="resume")
                self.reset()
        finally:
            if gen == self.gen:
                self.task = None

    async def do_barge(self, why: str):
        """The assistant stops; what the person says is heard on from its first syllable."""
        self.log(f"interruption ({why})")
        self.barge, self.mode, self.checked, self.barged = None, "listening", None, time.time()
        await self.send(e="barge")


async def serve(ws, transcribe, decide, clef_ok, log, lang="de"):
    """Run one WebSocket (already accepted) until it closes."""
    if not MODEL.exists():
        await ws.send_text(json.dumps({"e": "err", "text": "models/silero_vad.onnx missing"}))
        return await ws.close()
    ok = await asyncio.to_thread(clef_ok)
    ear = Ear(ws, transcribe, decide, clef_ok, lambda m: log(f"[ear] {m}"), lang)
    await ear.send(e="ready", clef=ok)
    log("[ear] connected" + (" (with Clef)" if ok else " (without Clef)"))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await ear.audio(msg["bytes"])
            elif msg.get("text"):
                try:
                    await ear.control(json.loads(msg["text"]))
                except ValueError:
                    pass
    except Exception as e:
        if type(e).__name__ != "WebSocketDisconnect":
            log(f"[ear] error: {e}")
    log("[ear] disconnected")
