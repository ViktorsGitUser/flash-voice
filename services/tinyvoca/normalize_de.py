"""German text normalization before eSpeak: spell out what a speaker reads aloud
differently from eSpeak's defaults (abbreviations, clock times, money, ordinals
before months, percent, years). Used for annotation, training and inference alike.
"""
import re

try:
    from num2words import num2words as _n2w
except ImportError:  # eSpeak still reads plain numbers; only the special forms need num2words
    _n2w = None

MONTHS = ("Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember")
ABBREVIATIONS = {
    "z.B.": "zum Beispiel", "z. B.": "zum Beispiel", "d.h.": "das heißt", "d. h.": "das heißt", "u.a.": "unter anderem",
    "usw.": "und so weiter", "bzw.": "beziehungsweise", "ca.": "circa", "vgl.": "vergleiche", "evtl.": "eventuell",
    "ggf.": "gegebenenfalls", "inkl.": "inklusive", "Nr.": "Nummer", "Dr.": "Doktor", "Prof.": "Professor",
    "Str.": "Straße", "Hr.": "Herr", "Fr.": "Frau", "St.": "Sankt", "etc.": "et cetera", "Mio.": "Millionen",
    "Mrd.": "Milliarden", "km/h": "Kilometer pro Stunde", "€": " Euro", "%": " Prozent",
}


def _num(value, to="cardinal"):
    if _n2w is None:
        return str(value)
    return _n2w(int(value), lang="de", to=to)


def _ordinal(match):
    word = _num(match[2], "ordinal")  # "dritte"
    prefix = match[1] or ""
    # After a preposition/article in the dative ("am", "vom", "zum", "bis zum") the ending is -n.
    if prefix.strip().lower() in ("am", "vom", "zum", "im", "ab dem", "bis zum", "seit dem", "dem", "den"):
        word += "n"
    return f"{prefix}{word} {match[3]}"


def _time(match):
    hours, minutes = int(match[1]), int(match[2])
    text = f"{_num(hours)} Uhr"
    if hours == 1:
        text = "ein Uhr"
    return text + (f" {_num(minutes)}" if minutes else "")


def _money(match):
    euros, cents = int(match[1]), int(match[2])
    text = f"{_num(euros)} Euro"
    return text + (f" {_num(cents)}" if cents else "")


def normalize(text):
    for short, long in ABBREVIATIONS.items():
        text = text.replace(short, long)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*Uhr\b", _time, text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time, text)
    text = re.sub(r"\b(\d+),(\d{2})\s*Euro\b", _money, text)
    text = re.sub(r"\b((?:[Aa]m|[Vv]om|[Zz]um|[Bb]is zum|[Ss]eit dem|[Aa]b dem) )?(\d{1,2})\. (" + "|".join(MONTHS) + r")\b", _ordinal, text)
    # Years after "im Jahr", or bare four-digit numbers 1100-1999 in running text, are read as years.
    text = re.sub(r"\b(1[1-9]\d\d)\b", lambda m: _num(m[1], "year"), text)
    text = re.sub(r"(\d+),(\d+)", lambda m: f"{_num(m[1])} Komma {' '.join(_num(d) for d in m[2])}", text)
    text = re.sub(r"(?<![\w-])-(\d+)", lambda m: "minus " + m[1], text)
    if _n2w is not None:
        text = re.sub(r"\d+", lambda m: _num(m[0]), text)
    text = re.sub(r"\bein(hundert|tausend)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
