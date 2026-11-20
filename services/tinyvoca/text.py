"""German front end (fork of the English tiny-voca front end): text -> phoneme tokens with a stress feature.

The same code runs during annotation, training-set preparation and inference,
so tokens can never drift between them. Two programs can produce the phonemes:

* ``espeak`` (front-end version 3, the default): eSpeak NG, as described below.
  The current checkpoints and annotations were made with it.
* ``own`` (front-end version 4): the separate ``dizione`` package, a dictionary
  built from CMUdict, a small model for unknown words and rules. No eSpeak, so
  the whole system can be shipped under a permissive licence. Same 76 symbols.

Choose with ``set_frontend("own")`` or ``TINY_VOCA_FRONTEND=own``. Everything
below about tokens holds for both; the rest of this note is the eSpeak path.

* eSpeak NG (``en-us``) is the only pronunciation source. It is an external
  rule-based program: no neural G2P and no downloaded weights.
* A token is one eSpeak phoneme exactly as eSpeak segments it (``--sep``), for
  example ``t``, ``aɪ``, ``ɑːɹ`` or ``n̩``. This is also the label inventory of
  the phoneme recogniser used for offline forced alignment, so one aligned
  label corresponds to exactly one token.
* Lexical stress is a parallel feature (0 none, 1 primary, 2 secondary), not a
  token, so it never has to be given audio frames.
* Punctuation, word boundaries and the utterance edges are explicit tokens that
  may last zero frames. They carry pauses when the recording has them.

Versions 1 and 2 encoded IPA as UTF-8 bytes and piped text to eSpeak without a
final newline; eSpeak NG 1.51 then drops the last character of the input, so
the final phonemes of every clause were wrong. Version 3 terminates the input
and has a regression test for it.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata

FRONTEND_VERSION = 103  # German fork: eSpeak NG "de", German phone inventory
ESPEAK_VOICE = "de"
SEPARATOR = "_"
FRONTENDS = ("espeak", "own")
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "espeak")
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
# German inventory: every phone is a label of the forced-alignment recogniser, except
# "ʊɐ", which eSpeak NG prints (and the recogniser was trained on) as "??".
CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ", "ç", "x", "h",
              "m", "n", "ŋ", "l", "ɾ", "r", "j", "ts", "pf", "tʃ", "dʒ")
VOWELS = ("ə", "ɐ", "ɜ", "ɪ", "i", "iː", "ɛ", "ɛː", "eː", "a", "ɑ", "ɑː", "ɔ", "oː", "ʊ", "uː",
          "y", "yː", "œ", "øː", "aɪ", "aʊ", "ɔø", "ʊɐ")
PHONES = CONSONANTS + VOWELS
SYMBOLS = SPECIALS + PUNCTUATION + PHONES
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
PHONE_SET = frozenset(PHONES)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES) | {SYMBOL_TO_ID[UNK]}
VOICELESS = frozenset(("p", "t", "k", "f", "s", "ʃ", "ç", "x", "h", "ts", "pf", "tʃ"))
VOWEL_SET = frozenset(VOWELS)
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

# eSpeak phonemes outside the inventory (English/French loan words, rare allophones).
FALLBACK = {
    "??": ("ʊɐ",), "ɹ": ("ɾ",), "ʁ": ("ɾ",), "w": ("v",), "ð": ("d",), "θ": ("s",), "ʔ": (),
    "e": ("eː",), "o": ("oː",), "u": ("uː",), "ø": ("øː",), "ɒ": ("ɔ",), "ʌ": ("a",), "æ": ("ɛ",),
    "ɜː": ("øː",), "ɔː": ("oː",), "ɪː": ("iː",), "aː": ("ɑː",), "eɪ": ("eː",), "ɛɪ": ("eː",),
    "əʊ": ("oː",), "oʊ": ("oː",), "eə": ("ɛː",), "ɪə": ("iː", "ɐ"), "ʊə": ("uː", "ɐ"), "aɪə": ("aɪ", "ɐ"),
    "ɔɪ": ("ɔø",), "ɔʏ": ("ɔø",), "əl": ("ə", "l"), "l̩": ("ə", "l"), "n̩": ("ə", "n"), "m̩": ("ə", "m"),
    "ɑ̃": ("ɑ", "ŋ"), "ɔ̃": ("ɔ", "ŋ"), "ɛ̃": ("ɛ", "ŋ"), "œ̃": ("œ", "ŋ"), "ʏ": ("y",), "ɚ": ("ɐ",),
    "ɲ": ("n", "j"), "ɬ": ("l",), "ɫ": ("l",), "dz": ("d", "z"), "β": ("v",), "ɣ": ("ɡ",), "ʎ": ("l", "j"),
}

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                         "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")"})
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
ABBREVIATIONS = {"dr", "prof", "nr", "str", "bzw", "ca", "usw", "vgl", "evtl", "ggf", "inkl", "hr", "fr", "st", "z", "b", "d", "h", "u", "a", "s", "o"}
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple  # eSpeak phonemes that had no inventory mapping

    def __len__(self):
        return len(self.ids)


def frontend():
    """Name of the program that produces the phonemes: ``"espeak"`` or ``"own"``."""
    return _frontend


def set_frontend(name):
    """Select the phonemizer for this process. Tokens of the two front ends
    must not be mixed: annotations and checkpoints record which one made them."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def _dizione():
    try:
        import dizione
    except ImportError as error:
        raise RuntimeError('the "own" front end needs the dizione package: pip install dizione') from error
    if tuple(dizione.symbols.SYMBOLS) != SYMBOLS:
        raise RuntimeError("dizione's symbol inventory differs from this model's")
    return dizione


def frontend_identity():
    if _frontend == "own":
        phonemizer = _dizione()
        return {"frontend_version": phonemizer.VERSION, "phonemizer": "dizione", "symbols": list(SYMBOLS)}
    return {"frontend_version": FRONTEND_VERSION, "voice": ESPEAK_VOICE, "separator": SEPARATOR,
            "symbols": list(SYMBOLS), "fallback": {key: list(value) for key, value in sorted(FALLBACK.items())}}


def frontend_runtime():
    """What an annotation run records about the exact program behind its
    tokens: the eSpeak version, or the fingerprint of the phonemizer build."""
    if _frontend == "own":
        return {"phonemizer": _dizione().identity()["fingerprint"]}
    return {"espeak_version": espeak_version()}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text):
    """Yield ``(is_punctuation, piece)``. Decimals, clock times, digit groups,
    contractions, title abbreviations and initialisms stay inside the text piece
    for eSpeak; every other punctuation mark becomes its own token."""
    start = 0
    for i, char in enumerate(text):
        if char not in _PUNCTUATION_CHARS:
            continue
        previous = text[i-1] if i else ""
        following = text[i+1] if i+1 < len(text) else ""
        if char == "'" and previous.isalnum() and following.isalnum():
            continue
        if char in ".,:" and previous.isdigit() and following.isdigit():
            continue
        if char == "." and previous.isdigit() and following == " " and i+2 < len(text) and text[i+2].isalnum():
            continue
        if char == ".":
            word = re.search(r"([A-Za-z]+)$", text[:i])
            if word and word[1].lower() in ABBREVIATIONS:
                continue
            if previous.isalpha() and following.isalpha():
                continue
        if text[start:i].strip():
            yield False, text[start:i].strip()
        yield True, char
        start = i+1
    if text[start:].strip():
        yield False, text[start:].strip()


def espeak_executable():
    executable = shutil.which("espeak-ng")
    if executable is None:
        raise RuntimeError("Install espeak-ng on the cloud machine before preprocessing/inference")
    return executable


def espeak_version():
    return subprocess.run([espeak_executable(), "--version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def espeak_words(piece):
    """Phonemize one text piece. Returns words as lists of ``(phone, stress)``.

    The input must end with a newline: eSpeak NG 1.51 silently drops the last
    character of unterminated standard input.
    """
    result = subprocess.run(
        [espeak_executable(), "-q", "--ipa", "--sep=" + SEPARATOR, "-v", ESPEAK_VOICE, "--stdin"],
        input=piece.rstrip("\n") + "\n", text=True, capture_output=True, check=True, timeout=30)
    return parse_espeak(result.stdout)


def parse_espeak(output):
    words = []
    for word in _LANGUAGE_TAG.sub("", output).split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def canonical(phone):
    """Map an eSpeak phoneme to inventory phones; ``None`` when unknown."""
    if phone in PHONE_SET:
        return (phone,)
    if phone in FALLBACK:
        return FALLBACK[phone]
    stripped = "".join(c for c in unicodedata.normalize("NFD", phone) if not unicodedata.combining(c))
    for candidate in (stripped, stripped.replace("ː", ""), stripped + "ː"):
        if candidate != phone and candidate in PHONE_SET:
            return (candidate,)
        if candidate != phone and candidate in FALLBACK:
            return FALLBACK[candidate]
    return None


def tokenize(pieces):
    """Build the token sequence from ``(is_punctuation, value)`` pieces where a
    text piece's value is the parsed eSpeak word list. Pure function (testable
    without eSpeak)."""
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:  # collapse repeats such as "!!"
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in PHONE_SET or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            for phone, level in word:
                mapped = canonical(phone)
                if mapped is None:
                    unknown.append(phone)
                    mapped = (UNK,)
                # A fallback may expand to several phones: stress goes to its first vowel.
                carrier = next((i for i, item in enumerate(mapped) if item in VOWEL_SET), 0)
                for index, item in enumerate(mapped):
                    symbols.append(item)
                    stress.append(level if index == carrier else 0)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in PHONE_SET or s == UNK for s in symbols):
        raise ValueError("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    if _frontend == "own":
        return tokenize(_dizione().pieces(text))
    pieces = []
    from .normalize_de import normalize as normalize_german, prepare
    for punctuation, piece in chunks(normalize_text(normalize_german(normalize_text(prepare(text))))):
        pieces.append((True, piece) if punctuation else (False, espeak_words(piece)))
    return tokenize(pieces)


def is_phone(token_ids):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    return [int(i) in PHONE_IDS for i in token_ids]
