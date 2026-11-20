"""The decider: the two questions the ear asks.
  finished(text, lang)        at the first pause - is what the person said a finished thought?   -> True/False/None
  barge(said, text, lang)     while the assistant speaks - what is the person doing?
                              -> {"interrupts": p, "listens": p, "other": p} or None

Three kinds (FV_DECIDER):
  mini   flash-turn, a small text model (English + German, ~70 MB, CPU, a few ms) - models/flash-turn/
  llm    a decision model in llama.cpp's llama-server (vllm-d3-mini, ~5 GB GPU) at FV_DECIDER_URL, /v1/systemone
  rules  none: pauses and simple word rules
  auto   (default) mini if its files are there, else llm if it answers, else rules
"""
import re
import time
from pathlib import Path

import numpy as np

from . import clients, config

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
PUNCT = re.compile(r"[.,;:!?…\"„“”‚‘’«»()\[\]–—-]+")


def _norm(text: str) -> str:
    """Punctuation removed, only a final "?" kept - as flash-turn was trained (speech recognition also writes full
    stops after cut-off phrases, but "?" only after questions)."""
    q = text.rstrip().endswith("?")
    t = " ".join(PUNCT.sub(" ", text).split())
    return t + "?" if q and t else t


def _texts(lang: str) -> dict:
    return TEXTS.get(lang, TEXTS["en"])


class Rules:
    name = "rules"

    def ok(self) -> bool:
        return False

    def finished(self, text, lang):
        return None

    def barge(self, said, text, lang):
        return None


class Llm:
    """A decision model behind llama-server's /v1/systemone (vllm-d3-mini)."""
    name = "llm"

    def ok(self) -> bool:
        return clients.decider_ok()

    def finished(self, text, lang):
        t = _texts(lang)
        # the bare sentence: wrapped in "The person says: ..." d3-mini took cut-off German phrases for finished ones
        # (6 of 24 test cases right, 23 of 24 with the bare sentence)
        r = clients.decide(text, {"f": {"type": "noul", "instructions": t["finished"]}})
        return r["answers"]["f"]["noul"] >= 0.6

    def barge(self, said, text, lang):
        t = _texts(lang)
        r = clients.decide(t["barge_state"].format(said=said[-400:], t=text), {"art": {
            "type": "choice", "instructions": t["barge_q"],
            "criteria": {k: t[k] for k in ("interrupts", "listens", "other")}}})
        return r["answers"]["art"]["probabilities"]


class Mini:
    """flash-turn: one small classifier for both questions (ONNX int8 on one CPU thread)."""
    name = "mini"

    def __init__(self, folder: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tok.enable_truncation(96, strategy="longest_first")
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(folder / "flash-turn.onnx"), sess_options=o, providers=["CPUExecutionProvider"])

    def ok(self) -> bool:
        return True

    def _p(self, first: str, text: str) -> np.ndarray:
        e = self.tok.encode(first, text)
        lg = self.s.run(None, {"input_ids": np.array([e.ids], np.int64),
                               "attention_mask": np.array([e.attention_mask], np.int64)})[0][0]
        return lg

    @staticmethod
    def _soft(x):
        x = np.exp(x - x.max())
        return x / x.sum()

    def finished(self, text, lang):
        p = self._soft(self._p("turn", _norm(text))[:2])
        return bool(p[1] >= 0.5)

    def barge(self, said, text, lang):
        said = " ".join(said.split())
        if len(said) > 160:
            said = "…" + said[-160:].split(" ", 1)[-1]
        p = self._soft(self._p("barge " + said, text)[2:])
        return {"interrupts": float(p[0]), "listens": float(p[1]), "other": float(p[2])}


_made = None


def get():
    """The decider to use (made once)."""
    global _made
    if _made is None:
        kind, folder = config.DECIDER, Path(config.TURN_MODEL)
        if kind in ("auto", "mini") and (folder / "flash-turn.onnx").is_file():
            t = time.time()
            _made = Mini(folder)
            print(f"[decider] flash-turn (mini, CPU) loaded in {time.time() - t:.1f} s", flush=True)
        elif kind in ("auto", "llm") and config.DECIDER_URL:
            _made = Llm()
        else:
            if kind == "mini":
                print(f"[decider] {folder}/flash-turn.onnx missing - rules only", flush=True)
            _made = Rules()
    return _made
