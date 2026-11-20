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

# English words that are common in German text, read the way German speakers say them (only sounds of the German
# inventory). Looked up without regard to case, after EXCEPTIONS. Words that are also German ("fast", "Post", "Bug",
# "Handy") are left out on purpose.
ENGLISH = {
    "Hugging": "h aˈ ɡ ɪ ŋ",
    "Face": "f eːˈ s",
    "HuggingFace": "h aˈ ɡ ɪ ŋ f eːˌ s",
    "GitHub": "ɡ ɪˈ t h a p",
    "GitLab": "ɡ ɪˈ t l ɛ p",
    "Software": "s ɔˈ f t v ɛː ɐ",
    "Hardware": "h ɑːˈ ɾ t v ɛː ɐ",
    "Laptop": "l ɛˈ p t ɔ p",
    "Browser": "b r aʊˈ z ɐ",
    "Update": "aˈ p d eː t",
    "Updates": "aˈ p d eː t s",
    "Download": "d aʊˈ n l oː t",
    "Downloads": "d aʊˈ n l oː t s",
    "Upload": "aˈ p l oː t",
    "Online": "ɔˈ n l aɪ n",
    "Offline": "ɔˈ f l aɪ n",
    "Meeting": "m iːˈ t ɪ ŋ",
    "Meetings": "m iːˈ t ɪ ŋ s",
    "Feedback": "f iːˈ t b ɛ k",
    "Chat": "tʃ ɛˈ t",
    "Chats": "tʃ ɛˈ t s",
    "chatten": "tʃ ɛˈ t ə n",
    "Chatbot": "tʃ ɛˈ t b ɔ t",
    "Chatbots": "tʃ ɛˈ t b ɔ t s",
    "Cloud": "k l aʊˈ t",
    "Website": "v ɛˈ p s aɪ t",
    "Websites": "v ɛˈ p s aɪ t s",
    "Smartphone": "s m ɑːˈ ɾ t f oː n",
    "Smartphones": "s m ɑːˈ ɾ t f oː n s",
    "Podcast": "p ɔˈ t k ɑː s t",
    "Streaming": "s t r iːˈ m ɪ ŋ",
    "Stream": "s t r iːˈ m",
    "Deadline": "d ɛˈ t l aɪ n",
    "Cool": "k uːˈ l",
    "Okay": "oː k eːˈ",
    "Design": "d iː z aɪˈ n",
    "Interface": "ɪˈ n t ɐ f eː s",
    "Feature": "f iːˈ tʃ ɐ",
    "Features": "f iːˈ tʃ ɐ s",
    "Release": "r iː l iːˈ s",
    "Coding": "k oːˈ d ɪ ŋ",
    "Developer": "d iː v ɛˈ l ɔ p ɐ",
    "Framework": "f r eːˈ m v œ ɾ k",
    "Workflow": "v œˈ ɾ k f l oː",
    "Startup": "s t ɑːˈ ɾ t a p",
    "Startups": "s t ɑːˈ ɾ t a p s",
    "Business": "b ɪˈ z n ɪ s",
    "Support": "s a p ɔˈ ɾ t",
    "Account": "ɛ k aʊˈ n t",
    "Login": "l ɔˈ ɡ ɪ n",
    "Password": "p ɑːˈ s v œ ɾ t",
    "Weekend": "v iːˈ k ɛ n t",
    "Event": "iː v ɛˈ n t",
    "Live": "l aɪˈ f",
    "Player": "p l eːˈ ɐ",
    "Game": "ɡ eːˈ m",
    "Games": "ɡ eːˈ m s",
    "Gaming": "ɡ eːˈ m ɪ ŋ",
    "Style": "s t aɪˈ l",
    "Hashtag": "h ɛˈ ʃ t ɛ k",
    "Tweet": "t v iːˈ t",
    "Likes": "l aɪˈ k s",
    "Follower": "f ɔˈ l oː ɐ",
    "Influencer": "ɪˈ n f l uː ɛ n s ɐ",
    "Content": "k ɔˈ n t ɛ n t",
    "Creator": "k r iː eːˈ t ɐ",
    "Speech": "s p iːˈ tʃ",
    "Text-to-Speech": "t ɛˈ k s t t uː s p iːˌ tʃ",
    "Speaker": "s p iːˈ k ɐ",
    "Headset": "h ɛˈ t s ɛ t",
    "Smartwatch": "s m ɑːˈ ɾ t v ɔ tʃ",
    "Display": "d ɪ s p l eːˈ",
    "Touch": "t aˈ tʃ",
    "Screen": "s k r iːˈ n",
    "Screenshot": "s k r iːˈ n ʃ ɔ t",
    "Backup": "b ɛˈ k a p",
    "Cache": "k ɛˈ ʃ",
    "Plugin": "p l aˈ ɡ ɪ n",
    "Upgrade": "aˈ p ɡ r eː t",
    "Training": "t r eːˈ n ɪ ŋ",
    "Know-how": "n oːˈ h aʊ",
    "Fairness": "f ɛːˈ ɐ n ɛ s",
    "Happy": "h ɛˈ p i",
    "Food": "f uːˈ t",
    "Burger": "b œˈ ɾ ɡ ɐ",
    "Sandwich": "s ɛˈ n t v ɪ tʃ",
    "Wellness": "v ɛˈ l n ɛ s",
    "Jogging": "dʒ ɔˈ ɡ ɪ ŋ",
    "Baby": "b eːˈ b i",
    "Babys": "b eːˈ b i s",
    "Sorry": "s ɔˈ r i",
    "Thanks": "t ɛˈ ŋ k s",
    "Please": "p l iːˈ s",
    "Hello": "h ɛ l oːˈ",
    "Bye": "b aɪˈ",
    "Party": "p ɑːˈ ɾ t i",
    "OpenAI": "oːˈ p ə n eː aɪˌ",
    "Google": "ɡ uːˈ ɡ ə l",
    "Apple": "ɛˈ p ə l",
    "Microsoft": "m aɪˈ k r oː s ɔ f t",
    "Amazon": "ɛˈ m ə z ɔ n",
    "YouTube": "j uːˈ t j uː p",
    "Facebook": "f eːˈ s b ʊ k",
    "WhatsApp": "v ɔˈ t s ɛ p",
    "Twitter": "t v ɪˈ t ɐ",
    "LinkedIn": "l ɪˈ ŋ k t ɪ n",
    "Spotify": "s p ɔˈ t ɪ f aɪ",
    "iPhone": "aɪˈ f oː n",
    "Android": "ɛˈ n d r ɔ ɪ t",
    "Windows": "v ɪˈ n d oː s",
    "ChatGPT": "tʃ ɛˈ t dʒ iː p iː t iːˌ",
    "Claude": "k l oːˈ t",
    "Anthropic": "ɛ n t r ɔˈ p ɪ k",
    "Python": "p aɪˈ t ɔ n",
    "JavaScript": "dʒ ɑːˈ v a s k r ɪ p t",
    "Netflix": "n ɛˈ t f l ɪ k s",
    "Instagram": "ɪˈ n s t a ɡ r ɛ m",
}
_ENGLISH_LOWER = {k.lower(): v for k, v in ENGLISH.items()}

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


def _kaffee(w, units):
    """"Kaffee" always with a long, stressed "ee" (as said in Austria; the unstressed eː came out short), also in
    compounds: Kaffeetasse -> main stress on the ee, Eiskaffee -> secondary stress on it."""
    if "kaffee" not in w.lower():
        return units
    phones = [u for u, _ in units]
    for i in range(len(phones) - 3):
        if phones[i:i + 4] == ["k", "a", "f", "eː"]:
            lead = w.lower().startswith("kaffee")
            out = [(u, 2 if lead and s == 1 else s) for u, s in units]
            out[i + 1] = ("a", 0)
            out[i + 3] = ("eː", 1 if lead else (units[i + 3][1] or 2))
            return out
    return units


def word(w, austrian=False):
    """One written word -> [(phone, stress), ...]."""
    return _kaffee(w, _word(w, austrian))


def _word(w, austrian=False):
    if austrian and w in AUSTRIAN:
        return _units(AUSTRIAN[w])
    if w in EXCEPTIONS:
        return _units(EXCEPTIONS[w])
    if w.lower() in _ENGLISH_LOWER:
        return _units(_ENGLISH_LOWER[w.lower()])
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
