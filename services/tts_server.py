"""flash-voice TTS service: Qwen3-TTS (Apache 2.0) through faster-qwen3-tts (MIT, static cache + CUDA graphs).

Two ways to get voices:
  * default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice - its 9 built-in speakers (serena, vivian, ono_anna, sohee,
    aiden, ryan, dylan, eric, uncle_fu), nothing else needed.
  * a voice library: FV_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base + FV_VOICES=<folder>. Every recording in it is a
    voice - name.wav (10-30 s of clean speech) + name.txt (exactly what is said), optional name.json (label,
    description, tags; "xvec_only": imitate only the colour of the voice, steadier). A mixed voice is only a speaker
    embedding (name.npy + its recipe in name.json). The library can be managed over HTTP (below). Only voices of
    people who agreed to it.

POST /speak         form: text, speaker, language (de|en|...), instruct, mix (JSON, preview a mixture), temperature
                    -> audio/wav
POST /speak_stream  same -> NDJSON, one {"audio": base64 WAV} per ~0.7 s (the first after ~0.4 s)
GET  /voices        [{"id", "label", "description", "tags", "origin", "studio", "xvec_only", "mix", "text", "created"}]
GET  /speakers      {id: label}          GET /health
library only:
POST /voices/clone  form: id, text, label, description, tags, clean, xvec_only; file: audio  -> a new recording
POST /voices/mix    form: id, parts (JSON [[voice, weight], ...]), label, description, tags, jitter, seed
POST /voices/update form: id, label, description, tags       POST /voices/delete  form: id
   (update/delete only voices made over this API: "studio": true)

Run:  python services/tts_server.py   (FV_TTS_PORT 8086, FV_TTS_HOST 127.0.0.1). ~5 GB of VRAM; on an RTX 5060 Ti the
1.7B model speaks ~2x faster than real time.
"""
import base64
import io
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from faster_qwen3_tts import FasterQwen3TTS

MODEL = os.environ.get("FV_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
PORT = int(os.environ.get("FV_TTS_PORT", "8086"))
HOST = os.environ.get("FV_TTS_HOST", "127.0.0.1")
LIBRARY = Path(os.environ.get("FV_VOICES", Path(__file__).resolve().parent.parent / "voices"))
CUSTOM = "customvoice" in MODEL.lower()
DEFAULT = os.environ.get("FV_TTS_DEFAULT", "serena")
BUILT_IN = {"serena": "Serena (f)", "vivian": "Vivian (f)", "ono_anna": "Anna (f)", "sohee": "Sohee (f)",
            "aiden": "Aiden (m)", "ryan": "Ryan (m)", "dylan": "Dylan (m)", "eric": "Eric (m)",
            "uncle_fu": "Uncle Fu (m, older)"}
# labels: a voice's own label (name.json), else FV_VOICES/names.json or FV_VOICE_NAMES (JSON {"id": "label"}),
# else the file name
NAMES = dict(BUILT_IN, **json.loads(os.environ.get("FV_VOICE_NAMES", "{}") or "{}"))
if (LIBRARY / "names.json").exists():   # the library's own labels ({"id": "label"}), e.g. in the speakers' language
    NAMES.update(json.loads((LIBRARY / "names.json").read_text(encoding="utf-8")))
LANGUAGES = {"de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
             "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
FRAMES_PER_S = 12        # the 12 Hz codec: one generation step per 1/12 s of sound
MAX_WAIT_S = 15          # a request that waited longer than this for the model: nobody is listening any more
sys.path.insert(0, str(Path(__file__).resolve().parent))


def library() -> dict:
    """{id: (wav path or None, text, options)} - every recording with its text and every mixed voice (name.npy)."""
    out = {}
    if not LIBRARY.exists():
        return out
    for f in sorted(list(LIBRARY.glob("*.wav")) + list(LIBRARY.glob("*.npy"))):
        txt, opt = f.with_suffix(".txt"), f.with_suffix(".json")
        options = json.loads(opt.read_text(encoding="utf-8")) if opt.exists() else {}
        if f.suffix == ".npy":
            if f.stem not in out:
                out[f.stem] = (None, "", dict(options, emb=str(f), xvec_only=True))
        elif txt.exists():
            out[f.stem] = (str(f), txt.read_text(encoding="utf-8").strip(), options)
    return out


app = FastAPI(title="flash-voice TTS")
_lock = threading.Lock()
print(f"[tts] loading {MODEL} ...", flush=True)
_model = FasterQwen3TTS.from_pretrained(MODEL, max_seq_len=1024)
VOICES = {v: (None, "", {}) for v in BUILT_IN} if CUSTOM else library()
if not VOICES:
    raise SystemExit(f"[tts] no voices: put name.wav + name.txt into {LIBRARY} (Base model) or use the CustomVoice model")


# ---------------------------------------------------------------- speaker embeddings (mixing voices, library only)
_emb_cache: dict = {}    # voice id -> its speaker embedding (torch, on the GPU)
_emb_like = None         # an embedding the model made: dtype and device for the ones read from .npy files


def _embedding(vid: str):
    """The speaker embedding (x-vector) of a voice: of its recording, or the saved one of a mixed voice. With _lock."""
    global _emb_like
    if vid in _emb_cache:
        return _emb_cache[vid]
    wav, _text, options = VOICES[vid]
    if options.get("emb"):
        if _emb_like is None:   # any recording tells the dtype/device
            first = next((v for v, (w, _t, _o) in VOICES.items() if w), None)
            if first is None:
                raise RuntimeError("no recording in the library")
            _embedding(first)
        recipe = options.get("mix") or {}
        parts = [p for p in recipe.get("parts") or [] if p[0] in VOICES and p[0] != vid]
        if parts and len(parts) == len(recipe["parts"]):
            # made again from its recipe with the model that runs now (0.6B and 1.7B embeddings differ in size)
            _emb_cache[vid] = emb = _mixed(parts, recipe.get("jitter", 0.0), recipe.get("seed", 0))
            return emb
        emb = torch.from_numpy(np.load(options["emb"])).to(device=_emb_like.device, dtype=_emb_like.dtype)
        if emb.shape != _emb_like.shape:
            raise RuntimeError(f"voice {vid} was mixed with another model and its recipe is missing")
    else:
        emb = _model.model.create_voice_clone_prompt(ref_audio=wav, ref_text="", x_vector_only_mode=True)[0].ref_spk_embedding
        _emb_like = emb
    _emb_cache[vid] = emb
    return emb


def _mixed(parts, jitter: float = 0.0, seed: int = 0):
    """A weighted mixture [[voice, weight], ...]; its length is the weighted mean length of the parts (a plain average
    of different voices is shorter than each and sounds dull). jitter > 0 moves it a bit (seed): a voice nearby."""
    parts = [(v, float(w)) for v, w in parts if v in VOICES and float(w) > 0]
    if not parts:
        raise ValueError("no valid voices in the mixture")
    total = sum(w for _v, w in parts)
    embs = [(_embedding(v).float(), w / total) for v, w in parts]
    mix = sum(e * w for e, w in embs)
    if jitter > 0:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(tuple(mix.shape), generator=g).to(mix.device)
        mix = mix + noise / noise.norm() * mix.norm() * float(jitter)
    length = sum(e.norm() * w for e, w in embs)
    return (mix / mix.norm() * length).to(dtype=_emb_like.dtype)


def _prompt(emb) -> dict:
    return dict(ref_code=[None], ref_spk_embedding=[emb], x_vector_only_mode=[True], icl_mode=[False])


def _clone_args(vid: str, mix=None) -> dict:
    """The model's arguments for a library voice: a recording (ref_audio + text) or an embedding. With _lock."""
    if mix:
        return {"voice_clone_prompt": _prompt(_mixed(mix["parts"], mix.get("jitter", 0.0), mix.get("seed", 0)))}
    wav, text, options = VOICES[vid]
    if options.get("emb"):
        return {"voice_clone_prompt": _prompt(_embedding(vid))}
    return {"ref_audio": wav, "ref_text": text, "xvec_only": bool(options.get("xvec_only"))}


def _stream(text: str, vid: str, language: str, steps: int, instruct: str = "", mix=None, temperature: float = 0.7):
    lang = LANGUAGES.get(language, "German")
    if CUSTOM:
        return _model.generate_custom_voice_streaming(text=text, speaker=vid, language=lang, instruct=instruct or None,
                                                      temperature=temperature, chunk_size=8, max_new_tokens=steps)
    return _model.generate_voice_clone_streaming(text=text, language=lang, instruct=instruct or None,
                                                 temperature=temperature, chunk_size=8, max_new_tokens=steps,
                                                 **_clone_args(vid, mix))


with _lock:   # read every recording once (the model keeps it) and record the CUDA graphs - before the first request
    for vid in (sorted(VOICES, key=lambda v: VOICES[v][0] is None) if not CUSTOM else []):   # recordings first (.npy dtype)
        try:
            list(_stream("Hallo.", vid, "de", 4 * FRAMES_PER_S))
            print(f"[tts]   voice {vid} read", flush=True)
        except Exception as e:   # a broken file must not keep the others from working
            print(f"[tts]   voice {vid} broken, left out: {e}", flush=True)
            VOICES.pop(vid, None)
    if DEFAULT not in VOICES:
        DEFAULT = next(iter(VOICES))
    for _ in range(2):   # the streamed way of speaking has its own first-time cost: pay it now
        list(_stream("Das Licht der Sonne sieht weiß aus, besteht aber aus allen Farben des Regenbogens.", DEFAULT, "de",
                     15 * FRAMES_PER_S))
torch.cuda.empty_cache()
print(f"[tts] ready on http://{HOST}:{PORT}, {len(VOICES)} voices ({'built-in' if CUSTOM else 'library ' + str(LIBRARY)}), "
      f"VRAM {torch.cuda.memory_reserved() / 1e9:.1f} GB", flush=True)


def _voice(speaker: str) -> str:
    return speaker if speaker in VOICES else DEFAULT


def _wav(audio, sr) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32).reshape(-1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# Even loudness: every sentence comes out of the model at its own level (an imitated voice up to 10 dB apart). A slow
# automatic gain per voice evens that out across pieces and sentences - the stress on single words stays.
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
            before = after = want   # a new sentence: the pause before it hides a jump
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
    """At most as much sound as the text can need (slow speech ~8 characters/s, plus 2 s): a model that does not stop
    after a short sentence would otherwise talk until its cache is full, and everything else waits."""
    return int(min(60, len(text) / 8 + 2) * FRAMES_PER_S)


def _generate(text, speaker, language, instruct, out: queue.Queue, cancel: threading.Event, asked: float, mix=None,
              temperature: float = 0.7):
    vid = _voice(speaker)
    level_key = "mix" if mix else vid
    t, total, pending, sr = None, 0.0, None, 24000
    try:
        with _lock:
            if time.time() - asked > MAX_WAIT_S or cancel.is_set():
                print(f"[tts] dropped: waited {time.time() - asked:.0f}s, nobody listens any more", flush=True)
                return
            t = time.time()
            for chunk, sr, _info in _stream(text, vid, language, _max_steps(text), instruct, mix, temperature):
                if cancel.is_set():
                    break
                audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
                # tiny pieces (a few ms at the end) go onto the next: Safari can hang on decoding such a crumb
                pending = np.concatenate([pending, audio]) if pending is not None else audio
                if len(pending) >= 0.12 * sr:
                    total += len(pending) / sr
                    try:
                        out.put((_level(pending, sr, level_key, first=total <= len(pending) / sr), sr), timeout=3)
                    except queue.Full:
                        cancel.set()
                    pending = None
            if pending is not None and len(pending) >= 0.03 * sr and not cancel.is_set():
                total += len(pending) / sr
                out.put((_level(pending, sr, level_key, first=total <= len(pending) / sr), sr), timeout=3)
            torch.cuda.empty_cache()
    except Exception as e:
        print("[tts] error:", e, flush=True)
    finally:
        out.put(None)
        if t:
            print(f"[tts] {time.time() - t:.2f}s for {total:.1f}s of speech (queued {t - asked:.1f}s)", flush=True)


def _pieces(text, speaker, language, instruct="", mix: str = "", temperature: float = 0.7):
    """The pieces as they come; when the caller stops reading (disconnect), the generation stops too."""
    out, cancel = queue.Queue(maxsize=8), threading.Event()
    mix = json.loads(mix) if mix and not CUSTOM else None
    temperature = min(1.2, max(0.3, float(temperature or 0.7)))
    threading.Thread(target=_generate, args=(text, speaker, language, instruct, out, cancel, time.time(), mix,
                                             temperature), daemon=True).start()
    try:
        while (item := out.get()) is not None:
            yield item
    finally:
        cancel.set()


@app.post("/speak")
def speak(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de"), instruct: str = Form(""),
          mix: str = Form(""), temperature: float = Form(0.7)):
    parts = list(_pieces(text, speaker, language, instruct, mix, temperature))
    if not parts:
        return JSONResponse({"error": "no speech made"}, status_code=503)
    return Response(_wav(np.concatenate([a for a, _ in parts]), parts[0][1]), media_type="audio/wav")


@app.post("/speak_stream")
def speak_stream(text: str = Form(...), speaker: str = Form(DEFAULT), language: str = Form("de"),
                 instruct: str = Form(""), mix: str = Form(""), temperature: float = Form(0.7)):
    def lines():
        for audio, sr in _pieces(text, speaker, language, instruct, mix, temperature):
            yield json.dumps({"audio": base64.b64encode(_wav(audio, sr)).decode()}) + "\n"
    return StreamingResponse(lines(), media_type="application/x-ndjson")


def _label(vid: str) -> str:
    return VOICES[vid][2].get("label") or NAMES.get(vid, vid.replace("_", " ").title())


@app.get("/speakers")
def speakers():
    return {vid: _label(vid) for vid in VOICES}


@app.get("/voices")
def voices():
    return [{"id": vid, "label": _label(vid), "description": options.get("description", ""),
             "tags": options.get("tags", []),
             "origin": "built-in" if CUSTOM else options.get("origin") or ("clone" if wav else "mix"),
             "studio": bool(options.get("studio")), "xvec_only": bool(options.get("xvec_only")),
             "mix": options.get("mix"), "text": text, "created": options.get("created")}
            for vid, (wav, text, options) in VOICES.items()]


# ---------------------------------------------------------------- managing the library (Base model only)
VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")


def _library_only():
    if CUSTOM:
        return JSONResponse({"error": "the built-in voices cannot be changed - a voice library needs the Base model "
                                      "(FV_TTS_MODEL=...-Base, FV_VOICES=folder)"}, status_code=400)
    return None


def _meta_form(label: str, description: str, tags: str) -> dict:
    return {"label": label.strip()[:60], "description": description.strip()[:500],
            "tags": [t.strip() for t in tags.split(",") if t.strip()][:12]}


def _check_new(vid: str):
    if (bad := _library_only()) is not None:
        return bad
    if not VOICE_ID.match(vid):
        return JSONResponse({"error": "Name: only a-z, 0-9, _ and - (2-41 characters)"}, status_code=400)
    if vid in VOICES or any((LIBRARY / f"{vid}.{ext}").exists() for ext in ("wav", "npy", "json")):
        return JSONResponse({"error": f"voice {vid} exists already"}, status_code=409)
    return None


def _own_voice(vid: str):
    if (bad := _library_only()) is not None:
        return bad
    if vid not in VOICES:
        return JSONResponse({"error": "unknown voice"}, status_code=404)
    if not VOICES[vid][2].get("studio"):
        return JSONResponse({"error": "only voices made over this API can be changed"}, status_code=403)
    return None


def _write_meta(vid: str, options: dict):
    keep = {k: v for k, v in options.items() if k != "emb"}
    (LIBRARY / f"{vid}.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1), encoding="utf-8")


@app.post("/voices/clone")
def voices_clone(audio: UploadFile = File(...), id: str = Form(...), text: str = Form(...), label: str = Form(""),
                 description: str = Form(""), tags: str = Form(""), clean: bool = Form(True),
                 xvec_only: bool = Form(False)):
    """A recording becomes a voice: any audio format (ffmpeg), at most 25 s, hiss and hum filtered (voice_filter.py).
    `text` must be exactly what is said."""
    import tempfile
    import voice_filter
    if (bad := _check_new(id)) is not None:
        return bad
    if len(text.strip()) < 10:
        return JSONResponse({"error": "the spoken text is missing"}, status_code=400)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.filename or "a.webm").suffix or ".webm") as f:
        f.write(audio.file.read())
        src = f.name
    try:
        raw = voice_filter.load(src)
    except Exception as e:
        return JSONResponse({"error": f"recording not readable: {e}"}, status_code=400)
    finally:
        os.unlink(src)
    sr = voice_filter.SR
    if len(raw) < 3 * sr:
        return JSONResponse({"error": "too short - at least 3 s, better 10-20 s of speech"}, status_code=400)
    raw = raw[: 25 * sr]
    out = voice_filter.clean(raw) if clean else raw / max(1e-6, float(np.abs(raw).max())) * 0.6
    options = dict(_meta_form(label or id, description, tags), origin="clone", studio=True,
                   xvec_only=bool(xvec_only), created=time.strftime("%Y-%m-%d %H:%M"))
    LIBRARY.mkdir(parents=True, exist_ok=True)
    sf.write(LIBRARY / f"{id}.wav", out.astype(np.float32), sr, subtype="PCM_16")
    (LIBRARY / f"{id}.txt").write_text(text.strip(), encoding="utf-8")
    _write_meta(id, options)
    with _lock:
        VOICES[id] = (str(LIBRARY / f"{id}.wav"), text.strip(), options)
        try:
            list(_stream("Hallo.", id, "de", 4 * FRAMES_PER_S))
        except Exception as e:
            VOICES.pop(id, None)
            for ext in ("wav", "txt", "json"):
                (LIBRARY / f"{id}.{ext}").unlink(missing_ok=True)
            return JSONResponse({"error": f"voice could not be read: {e}"}, status_code=500)
        finally:
            torch.cuda.empty_cache()
    print(f"[tts] new voice {id} (recording, {len(out) / sr:.1f} s)", flush=True)
    return {"id": id, "seconds": round(len(out) / sr, 1)}


@app.post("/voices/mix")
def voices_mix(id: str = Form(...), parts: str = Form(...), label: str = Form(""), description: str = Form(""),
               tags: str = Form(""), jitter: float = Form(0.0), seed: int = Form(0)):
    """Save a mixture of voices as a new voice (its embedding, name.npy)."""
    if (bad := _check_new(id)) is not None:
        return bad
    parts_list = json.loads(parts)
    jitter = min(0.5, max(0.0, float(jitter)))
    with _lock:
        try:
            emb = _mixed(parts_list, jitter, seed)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        np.save(LIBRARY / f"{id}.npy", emb.float().cpu().numpy())
        options = dict(_meta_form(label or id, description, tags), origin="mix", studio=True, xvec_only=True,
                       mix={"parts": parts_list, "jitter": jitter, "seed": seed}, created=time.strftime("%Y-%m-%d %H:%M"))
        _write_meta(id, options)
        VOICES[id] = (None, "", dict(options, emb=str(LIBRARY / f"{id}.npy")))
        _emb_cache[id] = emb
    print(f"[tts] new voice {id} (mixture {parts_list}, jitter {jitter})", flush=True)
    return {"id": id}


@app.post("/voices/update")
def voices_update(id: str = Form(...), label: str = Form(""), description: str = Form(""), tags: str = Form("")):
    if (bad := _own_voice(id)) is not None:
        return bad
    wav, text, options = VOICES[id]
    options = dict(options, **_meta_form(label or options.get("label") or id, description, tags))
    _write_meta(id, options)
    VOICES[id] = (wav, text, options)
    return {"id": id}


@app.post("/voices/delete")
def voices_delete(id: str = Form(...)):
    if (bad := _own_voice(id)) is not None:
        return bad
    with _lock:
        wav, _text, _options = VOICES.pop(id)
        _emb_cache.pop(id, None)
        if wav:   # the model keeps what it read from a recording
            for key in [k for k in getattr(_model, "_voice_prompt_cache", {}) if k[0] == wav]:
                _model._voice_prompt_cache.pop(key, None)
    for ext in ("wav", "txt", "json", "npy"):
        (LIBRARY / f"{id}.{ext}").unlink(missing_ok=True)
    _gain_db.pop(id, None)
    print(f"[tts] voice {id} deleted", flush=True)
    return {"id": id}


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "voices": len(VOICES), "library": None if CUSTOM else str(LIBRARY)}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
