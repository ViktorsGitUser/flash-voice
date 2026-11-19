"""apptor-tiny-tts German letter-to-sound front end without eSpeak NG (Apache 2.0).

A small model (convolutions + bidirectional GRU, CTC, ~3 MB ONNX) turns each written word into phones with stress,
in the same inventory and conventions as eSpeak NG "de", which the speech models were trained on. It was trained
on the eSpeak readings of ~480,000 words from the Common Voice German sentences, so it reproduces those readings
without needing the eSpeak program (GPL-3.0) at run time. A small hand-written list (EXCEPTIONS) wins over the model.

    words(piece) -> [[(phone, stress), ...], ...]      # drop-in for tiny_voca.text.espeak_words
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = Path(os.environ.get("APPTOR_G2P_MODEL", HERE / "apptor-g2p-de.onnx"))
MARKS = {"ˈ": 1, "ˌ": 2}

# Hand-written readings (word -> phones with stress marks, space separated), used before the model: mostly first
# names, where the eSpeak readings (and so the model) put the stress on the wrong syllable.
EXCEPTIONS = {
    "Helene": "h eː l eːˈ n ə",
    "Diana": "d i aˈ n ɑː",
    "Dianna": "d i aˈ n ɑː",
    "Daniela": "d a n i eːˈ l ɑː",
    "Melanie": "m eːˈ l a n iː",
    "Valentina": "v a l ɛ n t iːˈ n ɑː",
    "Theresa": "t eː r eːˈ z ɑː",
    "Elisabeth": "eː l iːˈ z a b ɛ t",
    "Alexander": "a l ɛ k s aˈ n d ɜ",
    "Andreas": "a n d r eːˈ a s",
    "Matthias": "m a t iːˈ a s",
    "Tobias": "t oː b iːˈ a s",
    "Florian": "f l oːˈ r i a n",
    "Sebastian": "z eː b aˈ s t i a n",
    "Stefanie": "ʃ t eːˈ f a n iː",
    "Christina": "k r ɪ s t iːˈ n ɑː",
    "Magdalena": "m a k d a l eːˈ n ɑː",
    "Veronika": "v eː r oːˈ n i k ɑː",
    "Monika": "m oːˈ n i k ɑː",
    "Angelika": "a ŋ ɡ eːˈ l i k ɑː",
    "Leonie": "l eː oːˈ n iː",
    "Emilia": "eː m iːˈ l i ɑː",
    "David": "d ɑːˈ v ɪ t",
    "Raphael": "r ɑːˈ f a eː l",
    "Michael": "m ɪˈ ç a eː l",
    "Manuel": "m ɑːˈ n uː ɛ l",
    "Maximilian": "m a k s i m iːˈ l i ɑː n",
    "Julia": "j uːˈ l i ɑː",
    "Sarah": "z ɑːˈ r ɑː",
    "Barbara": "b aˈ ɾ b a r ɑː",
    "Gabriele": "ɡ a b r i eːˈ l ə",
    "Ursula": "ʊˈ ɾ z uː l ɑː",
    "Irene": "iː r eːˈ n ə",
    "Miriam": "m iːˈ r i a m",
    "Marlene": "m a ɾ l eːˈ n ə",
    "Helena": "h eːˈ l eː n ɑː",
    "Elena": "eːˈ l eː n ɑː",
    "Selina": "z eː l iːˈ n ɑː",
    "Martina": "m a ɾ t iːˈ n ɑː",
    "Claudia": "k l aʊˈ d i ɑː",
    "Ingrid": "ɪˈ ŋ ɡ r ɪ t",
    "Leopold": "l eːˈ oː p ɔ l t",
    "Ferdinand": "f ɛˈ ɾ d i n a n t",
    "Dominik": "d oːˈ m i n ɪ k",
    "Fabian": "f ɑːˈ b i a n",
    "Kilian": "k iːˈ l i a n",
    "Elias": "eː l iːˈ a s",
    "Noah": "n oːˈ a",
    "Anton": "aˈ n t oː n",
    "Dorothea": "d oː r oː t eːˈ ɑː",
    "Cornelia": "k ɔ ɾ n eːˈ l i ɑː",
    "Philomena": "f i l oː m eːˈ n ɑː",
    "Aurelia": "aʊ r eːˈ l i ɑː",
    "Antonia": "a n t oːˈ n i ɑː",
    "Paulina": "p aʊ l iːˈ n ɑː",
    "Juliana": "j uː l i ɑːˈ n ɑː",
    "Liliana": "l i l i ɑːˈ n ɑː",
    "Adriana": "a d r i ɑːˈ n ɑː",
    "Tatjana": "t a t j ɑːˈ n ɑː",
    "Svetlana": "s v ɛ t l ɑːˈ n ɑː",
    "Milena": "m i l eːˈ n ɑː",
    "Daria": "d ɑːˈ r i ɑː",
    "Mariana": "m a r i ɑːˈ n ɑː",
    "Ariana": "a r i ɑːˈ n ɑː",
}

# Austrian standard pronunciation (Österreichisches Wörterbuch) where the German German reading differs: used only
# when the Austrian accent is on (austrian=True), so the Hochdeutsch voices keep the German reading.
AUSTRIAN = {
    "Kaffee": "k a f eːˈ", "Kaffees": "k a f eːˈ s", "Kaffeehaus": "k a f eːˈ h aʊˌ s",
    "Kaffeehäuser": "k a f eːˈ h ɔøˌ z ɜ", "Tabak": "t a b aˈ k", "Mathematik": "m a t eːˈ m a t ɪ k",
    "Chemie": "k eː m iːˈ", "Chemiker": "k eːˈ m ɪ k ɜ", "Chemikalien": "k eːˌ m iː k ɑˈ l ɪ ə n",
    "chemisch": "k eːˈ m ɪ ʃ", "chemische": "k eːˈ m ɪ ʃ ə", "chemischen": "k eːˈ m ɪ ʃ ə n",
    "China": "k iːˈ n ɑː", "Chinese": "k iː n eːˈ z ə", "Chinesen": "k iː n eːˈ z ə n",
    "chinesisch": "k iː n eːˈ z ɪ ʃ", "chinesische": "k iː n eːˈ z ɪ ʃ ə", "Chirurg": "k iː r ??ˈ k",
    "Motor": "m oː t oːˈ ɾ", "Platin": "p l a t iːˈ n", "Billard": "b ɪ j ɑːˈ ɾ", "Garage": "ɡ a r ɑːˈ ʃ",
    # western-Austrian "scht" crept into the accent here; lenis d (as said in Vienna) keeps a clean s
    "Donnerstag": "d ɔˈ n ɐ s d ɑː k", "Donnerstags": "d ɔˈ n ɐ s d ɑː k s",
}

_state = {}


def _load():
    if "session" not in _state:
        import onnxruntime as ort
        meta = json.loads(MODEL.with_suffix(".json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        _state["session"] = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
        _state["chars"] = {c: i for i, c in enumerate(meta["chars"])}
        _state["units"] = meta["units"]
        _state["up"] = meta["up"]
        _state["max_len"] = meta["max_len"]
        _state["cache"] = {}
    return _state


def _units(text):
    out = []
    for unit in text.split():
        stress = 0
        for mark, level in MARKS.items():
            if mark in unit:
                stress = stress or level
                unit = unit.replace(mark, "")
        if unit:
            out.append((unit, stress))
    return out


def word(w, austrian=False):
    """One written word -> [(phone, stress), ...]."""
    if austrian and w in AUSTRIAN:
        return _units(AUSTRIAN[w])
    if w in EXCEPTIONS:
        return _units(EXCEPTIONS[w])
    s = _load()
    cache = s["cache"]
    if w in cache:
        return cache[w]
    if len(w) > s["max_len"]:   # very long compound: read it in parts
        half = len(w) // 2
        result = word(w[:half]) + word(w[half:])
    else:
        ids = np.array([[s["chars"].get(c, 1) for c in w]], dtype=np.int64)
        logp = s["session"].run(None, {"chars": ids})[0][0]
        blank = len(s["units"])
        best = logp.argmax(-1)
        units, prev = [], None
        for b in best:
            if b != prev and b != blank:
                units.append(s["units"][b])
            prev = b
        result = _units(" ".join(units))
    if len(cache) < 100000:
        cache[w] = result
    return result


def words(piece, austrian=False):
    """A text piece (normalized, no punctuation) -> words as lists of (phone, stress), like espeak_words."""
    return [u for u in (word(w, austrian) for w in piece.split()) if u]


def phonemize(text, austrian=False):
    """Text -> tokens like tiny_voca.text.phonemize, with this model instead of eSpeak NG. ``austrian``: use the
    Austrian readings (AUSTRIAN) - for the accent switch of apptor-tiny-tts-at."""
    try:
        from tiny_voca.normalize_de import normalize as normalize_german
        from tiny_voca.text import chunks, normalize_text, tokenize
    except ImportError:   # flash-voice vendors the front end as "tinyvoca"
        from tinyvoca.normalize_de import normalize as normalize_german
        from tinyvoca.text import chunks, normalize_text, tokenize
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    pieces = [(True, p) if punct else (False, words(p, austrian))
              for punct, p in chunks(normalize_text(normalize_german(normalize_text(text))))]
    return tokenize(pieces)
