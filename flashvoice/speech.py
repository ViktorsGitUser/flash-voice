"""Text for the voice: what is not read out (code, links, markdown, emojis), and where a streamed answer is cut into
pieces that are voiced one after the other. Where a text is cut decides the intonation: at the end of a sentence,
a long sentence at a natural pause near its middle - never inside a clause.
"""
import re

# a point that ends no sentence: a date or ordinal ("der 8. Oktober"), a single letter ("z. B.", "e. g."), an abbreviation
NO_END = re.compile(r"(?:\b\d{1,2}|\b\w|\b(?:bzw|ca|Dr|Prof|St|Nr|usw|vgl|etc|evtl|ggf|inkl|Mio|Mrd|Jh|Hr|Fr|Abb|"
                    r"bspw|sog|ugs|Mr|Mrs|Ms|vs|approx|incl|no|fig))\.$", re.I)
_END = re.compile(r"[.!?…:;](\s+)")
FIRST_PART = re.compile(r"(.{25,}?,)(\s+)", re.S)   # the very first piece may end at a comma: heard sooner
SPEECH_MAX, SPEECH_MIN = 200, 25
THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
FENCE = "`" * 3


def clean(text: str) -> str:
    """Markdown answer -> what a voice says."""
    text = re.sub(FENCE + r".*?(" + FENCE + "|$)", " ", text, flags=re.S)   # code is not read out
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)                   # [link](url) -> link
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", text, flags=re.M)        # list marks
    text = re.sub(r"[*_#`>~|]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", "", text)  # emojis
    return re.sub(r"\s+", " ", text).strip()


def first_sentence(buf: str):
    """(sentence, where the rest starts) for the first complete sentence in a streamed text, or None. Only a sign
    followed by a space ends one: a point at the very end of a streamed piece may belong to 'example.com' or
    '8. Oktober'."""
    for m in _END.finditer(buf):
        sentence = buf[:m.start() + 1]
        if buf[m.start()] == "." and NO_END.search(sentence):
            continue
        return sentence, m.end()
    return None


def _clause_split(s: str) -> list[str]:
    if len(s) <= SPEECH_MAX:
        return [s]
    best, score = None, None
    for m in re.finditer(r"(?:;|:| – | - |,)\s+", s):
        cut = m.end()
        if cut < SPEECH_MIN or len(s) - cut < SPEECH_MIN:
            continue
        weight = {";": 0, ":": 0}.get(m.group(0).strip()[:1], 1)   # a semicolon/colon pauses more than a comma
        sc = abs(cut - len(s) / 2) + weight * 20
        if score is None or sc < score:
            best, score = cut, sc
    if best is None:
        sp = s.rfind(" ", 0, len(s) // 2 + 20)
        best = sp + 1 if sp > SPEECH_MIN else None
    if best is None:
        return [s]
    return _clause_split(s[:best].strip()) + _clause_split(s[best:].strip())


class Splitter:
    """Feed it the streamed answer; it hands back the pieces to voice as soon as they are complete. A code block is
    shown on the page, not spoken; a heading or list line is a piece of its own."""
    def __init__(self):
        self.buf, self.first = "", True

    def _next(self, work: str):
        """(piece, characters used) or None while the next piece is not complete yet."""
        line_end = work.find("\n")
        m = first_sentence(work)
        if line_end >= 0 and (m is None or line_end < len(m[0])):
            piece = work[:line_end]
            if re.match(r"^\s*#", piece) and clean(piece) and not re.search(r"[.!?:]$", clean(piece)):
                piece += "."   # a heading is spoken as a short phrase with a pause after it
            return piece, line_end + 1
        if m is None and self.first:
            fm = FIRST_PART.match(work)
            m = (fm.group(1), fm.end()) if fm else None
        if m is None or (len(m[0]) < 12 and len(work) < 60):
            return None
        return m

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while True:
            self.buf = re.sub(FENCE + ".*?" + FENCE, "\n", self.buf, flags=re.S)   # finished code blocks: away
            open_at = self.buf.find(FENCE)
            got = self._next(self.buf if open_at < 0 else self.buf[:open_at])
            if got is None:
                if open_at > 0:   # the text before a code block that is still being written
                    out += self._take(self.buf[:open_at])
                    self.buf = self.buf[open_at:]
                break
            out += self._take(got[0])
            self.buf, self.first = self.buf[got[1]:], False
        return out

    def _take(self, piece: str) -> list[str]:
        t = clean(piece)
        return _clause_split(t) if t else []

    def flush(self) -> list[str]:
        rest, self.buf = re.sub(FENCE + ".*", " ", self.buf, flags=re.S), ""
        return self._take(rest)
