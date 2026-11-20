"""Mathematics read aloud in German: formulas, LaTeX and symbols -> words, before the usual normalization.

"ζ(s) = 0 für Re(s) = 1/2"  -> "Zeta von s gleich 0 für Realteil von s gleich ein halb"
"$\\sum_{n=1}^{\\infty} n^{-s}$" -> "Summe von n gleich 1 bis unendlich über n hoch minus s"
Numbers stay digits here (normalize_de spells them out afterwards). Plain prose is left alone: a "-" is read as
"minus" only between numbers, single letters or brackets, "/" as "durch" only between numbers or single letters.
"""
import re

GREEK = {
    "alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "delta": "Delta", "epsilon": "Epsilon", "varepsilon": "Epsilon",
    "zeta": "Zeta", "eta": "Eta", "theta": "Theta", "vartheta": "Theta", "iota": "Iota", "kappa": "Kappa",
    "lambda": "Lambda", "mu": "Mü", "nu": "Nü", "xi": "Xi", "omicron": "Omikron", "pi": "Pi", "rho": "Rho",
    "sigma": "Sigma", "tau": "Tau", "upsilon": "Ypsilon", "phi": "Phi", "varphi": "Phi", "chi": "Chi", "psi": "Psi",
    "omega": "Omega",
}
GREEK_CHARS = dict(zip("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩϑϕϵ",
                       ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda",
                        "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "sigma", "tau", "upsilon", "phi", "chi", "psi",
                        "omega", "gamma", "delta", "theta", "lambda", "xi", "pi", "sigma", "phi", "psi", "omega",
                        "theta", "phi", "epsilon"]))
# LaTeX commands and symbols -> words (spaces around, collapsed later)
WORDS = {
    r"\infty": "unendlich", "∞": "unendlich", r"\to": "gegen", r"\rightarrow": "gegen", "→": "gegen",
    r"\Rightarrow": "daraus folgt", "⇒": "daraus folgt", r"\Leftrightarrow": "genau dann wenn", "⇔": "genau dann wenn",
    r"\cdot": "mal", "·": "mal", "⋅": "mal", r"\times": "mal", "×": "mal", r"\div": "durch", "÷": "durch",
    r"\leq": "kleiner gleich", r"\le": "kleiner gleich", "≤": "kleiner gleich",
    r"\geq": "größer gleich", r"\ge": "größer gleich", "≥": "größer gleich",
    r"\neq": "ungleich", r"\ne": "ungleich", "≠": "ungleich", r"\approx": "ungefähr", "≈": "ungefähr",
    r"\equiv": "identisch", "≡": "identisch", r"\sim": "ungefähr", r"\pm": "plus minus", "±": "plus minus",
    r"\in": "Element von", "∈": "Element von", r"\notin": "nicht Element von", "∉": "nicht Element von",
    r"\subset": "Teilmenge von", "⊂": "Teilmenge von", r"\cup": "vereinigt mit", "∪": "vereinigt mit",
    r"\cap": "geschnitten mit", "∩": "geschnitten mit", r"\forall": "für alle", "∀": "für alle",
    r"\exists": "es gibt", "∃": "es gibt", r"\partial": "d", "∂": "d", r"\nabla": "Nabla",
    r"\ldots": "und so weiter", r"\dots": "und so weiter", r"\cdots": "und so weiter", "…": "und so weiter",
    r"\mathbb{N}": "die natürlichen Zahlen", "ℕ": "die natürlichen Zahlen", r"\mathbb{Z}": "die ganzen Zahlen",
    "ℤ": "die ganzen Zahlen", r"\mathbb{Q}": "die rationalen Zahlen", "ℚ": "die rationalen Zahlen",
    r"\mathbb{R}": "die reellen Zahlen", "ℝ": "die reellen Zahlen", r"\mathbb{C}": "die komplexen Zahlen",
    "ℂ": "die komplexen Zahlen", r"\Re": "Realteil", r"\Im": "Imaginärteil", r"\ln": "Logarithmus naturalis",
    r"\log": "Logarithmus", r"\exp": "e hoch", r"\sin": "Sinus", r"\cos": "Kosinus", r"\tan": "Tangens",
    r"\lim": "Limes", r"\max": "Maximum", r"\min": "Minimum", r"\gcd": "größter gemeinsamer Teiler", "√": "Wurzel aus",
    "∑": "Summe", "∏": "Produkt", "∫": "Integral", "−": "-", "≪": "viel kleiner als", "≫": "viel größer als",
}
BIG = {"sum": "Summe", "prod": "Produkt", "int": "Integral", "lim": "Limes"}
FUNCS = {"Re": "Realteil", "Im": "Imaginärteil", "ln": "Logarithmus naturalis", "log": "Logarithmus", "exp": "e hoch",
         "sin": "Sinus", "cos": "Kosinus", "tan": "Tangens", "li": "Integrallogarithmus", "Li": "Integrallogarithmus"}
FRACTIONS = {(1, 2): "ein halb", (1, 3): "ein Drittel", (2, 3): "zwei Drittel", (1, 4): "ein Viertel",
             (3, 4): "drei Viertel", (1, 5): "ein Fünftel", (1, 8): "ein Achtel", (1, 10): "ein Zehntel"}
LETTERS = {"a": "a", "b": "be", "c": "ze", "d": "de", "e": "e", "f": "eff", "g": "ge", "h": "ha", "i": "i", "j": "jott",
           "k": "ka", "l": "ell", "m": "emm", "n": "enn", "o": "o", "p": "pe", "q": "ku", "r": "err", "s": "ess",
           "t": "te", "u": "u", "v": "vau", "w": "we", "x": "iks", "y": "üpsilon", "z": "zett"}
SUPER = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿⁱ", "0123456789+-ni"))
SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋ₙᵢ", "0123456789+-ni"))
_ARG = r"\{((?:[^{}]|\{[^{}]*\})*)\}"   # one {...} argument (one level of nested braces)
_OPERAND = r"(?:(?<=\d)|(?<=\))|(?<=(?<![\w])[A-Za-z]))"   # left of an operator: a number, ")", a single letter


def _latex(t):
    t = re.sub(r"\\(?:text|mathrm|mathit|operatorname|mathbf|textbf)\s*" + _ARG, r" \1 ", t)
    t = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b\s*", "", t)
    t = re.sub(r"\\[,;:!]|\\quad|\\qquad|~", " ", t)
    for _ in range(3):   # nested fractions and roots, innermost first
        t = re.sub(r"\\[dt]?frac\s*" + _ARG + r"\s*" + _ARG, lambda m: _fraction(m[1], m[2]), t)
        t = re.sub(r"\\sqrt\s*\[([^\]]+)\]\s*" + _ARG, r" \1-te Wurzel aus ( \2 ) ", t)
        t = re.sub(r"\\sqrt\s*" + _ARG, r" Wurzel aus ( \1 ) ", t)
    # \sum_{n=1}^{\infty}, \int_0^1, \lim_{x \to 0}
    lim = r"(?:_\s*(?:\{([^{}]*)\}|(\S))\s*)?(?:\^\s*(?:\{([^{}]*)\}|(\S))\s*)?"
    def big(m):
        name, low, high = BIG[m[1]], m[2] or m[3], m[4] or m[5]
        if m[1] == "lim":
            return f" {name} für {low} " if low else f" {name} "
        out = f" {name}"
        if low:
            out += f" von {low}"
        if high:
            out += f" bis {high}"
        return out + " über "
    t = re.sub(r"\\(sum|prod|int|lim)(?![A-Za-z])\s*" + lim, big, t)
    for cmd, word in sorted(WORDS.items(), key=lambda kv: -len(kv[0])):
        if cmd.startswith("\\"):
            t = re.sub(re.escape(cmd) + r"(?![A-Za-z])", f" {word} ", t)
    t = re.sub(r"\\([A-Za-z]+)", lambda m: f" {GREEK.get(m[1], m[1])} ", t)   # \zeta, unknown commands: the name
    return t


def _fraction(a, b):
    a, b = a.strip(), b.strip()
    if a.isdigit() and b.isdigit() and (int(a), int(b)) in FRACTIONS:
        return f" {FRACTIONS[int(a), int(b)]} "
    simple = lambda x: re.fullmatch(r"[^\s+\-]+", x)   # n^s: no brackets needed when listening
    return f" {a if simple(a) else '( ' + a + ' )'} durch {b if simple(b) else '( ' + b + ' )'} "


def _power(m):
    base, exp = m[1], (m[2] if m[2] is not None else m[3]).strip()
    if exp == "2":
        return f"{base} Quadrat"
    if exp == "3":
        return f"{base} hoch drei"
    return f"{base} hoch {exp}" + (" " if len(exp.split()) <= 2 else " Ende Exponent ")


def speak_math(text):
    if not re.search(r"[\\^_=+<>/$∑∏∫√∞≤≥≠≈±·×÷→⇒∈ℕℤℚℝℂ⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉" + "".join(GREEK_CHARS) + "]", text):
        return text
    t = text
    # LaTeX delimiters: $$..$$, $..$, \( .. \), \[ .. \]
    t = re.sub(r"\$\$(.+?)\$\$", lambda m: " " + _latex(m[1]) + " ", t, flags=re.S)
    t = re.sub(r"(?<![\w$])\$([^$\n]{1,300}?)\$(?!\d)", lambda m: " " + _latex(m[1]) + " ", t)
    t = re.sub(r"\\\((.+?)\\\)|\\\[(.+?)\\\]", lambda m: " " + _latex(m[1] or m[2]) + " ", t, flags=re.S)
    if "\\" in t:
        t = _latex(t)
    t = "".join(f" {GREEK[GREEK_CHARS[c]]} " if c in GREEK_CHARS else c for c in t)
    t = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿⁱ]+", lambda m: "^{" + "".join(SUPER[c] for c in m[0]) + "}", t)
    t = re.sub(r"[₀₁₂₃₄₅₆₇₈₉₊₋ₙᵢ]+", lambda m: "_{" + "".join(SUB[c] for c in m[0]) + "}", t)
    for sym, word in sorted(WORDS.items(), key=lambda kv: -len(kv[0])):
        if not sym.startswith("\\"):
            t = t.replace(sym, f" {word} ")
    # powers and indices: x^2, n^{-s}, 10^12, a_n, a_{n+1}
    t = re.sub(r"(\w|\))\s*\^\s*(?:\{([^{}]*)\}|(-?\w+))", _power, t)
    t = re.sub(r"(?<=\w)_\s*(?:\{([^{}]*)\}|(\w))", lambda m: " " + (m[1] or m[2]) + " ", t)
    # functions: ζ(s) -> Zeta von s, Re(s) -> Realteil von s, f(x) -> f von x
    names = "|".join(sorted(list(GREEK.values()) + list(FUNCS), key=len, reverse=True))
    t = re.sub(rf"(?<![\w])(?:({names})\s*|([A-Za-z]))\(([^()]{{1,40}})\)",
               lambda m: f"{FUNCS.get(m[1] or m[2], m[1] or m[2])} von {m[3]}", t)
    t = re.sub(r"(?<![\w\\])(log|ln|sin|cos|tan|exp)(?!\w)", lambda m: FUNCS[m[1]], t)
    if "Integral" in t:   # dx, dt after an integral
        t = re.sub(r"(?<![\w])d([a-z])(?![\w])", r"de \1", t)
    t = re.sub(r"(?<![\d/])24\s*/\s*7(?![\d/])", "vierundzwanzig sieben", t)
    t = re.sub(r"(?:(?<=\d)|(?<=(?<![\w])[A-Za-z]))!(?=\s|$|[.,;:)])", " Fakultät", t)
    # 1/2 + it (complex numbers on the critical line): "i t", not the English "it"
    t = re.sub(r"(?<=[\d\s])([+-])\s*i([a-z])\b", r" \1 i mal \2", t)
    # fractions of numbers, then the operators
    t = re.sub(r"(?<![\d/])(\d+)\s*/\s*(\d+)(?![\d/])",
               lambda m: f" {FRACTIONS[int(m[1]), int(m[2])]} " if (int(m[1]), int(m[2])) in FRACTIONS
               else f" {m[1]} durch {m[2]} ", t)
    t = re.sub(_OPERAND + r"\s*/\s*(?=[\w(])", " durch ", t)
    t = re.sub(r"\s*=\s*", " gleich ", t)
    t = re.sub(r"(?<=[\w)])\s*\+\s*(?=[\w(])", " plus ", t)
    t = re.sub(r"(?<=[\s(])\+(?=\w)", " plus ", t)
    t = re.sub(_OPERAND + r"\s*-\s*(?=[\w(])(?![A-Za-zÄÖÜäöüß]{2})", " minus ", t)
    t = re.sub(_OPERAND + r"\s*\*\s*(?=[\w(])", " mal ", t)
    t = re.sub(r"(?<=[\w)])\s*<\s*(?=[\w(-])", " kleiner als ", t)
    t = re.sub(r"(?<=[\w)])\s*>\s*(?=[\w(-])", " größer als ", t)
    t = re.sub(r"(?<=[\s(])-(?=[A-Za-z]\b)", "minus ", t)
    # brackets inside a formula: read only when they group something
    t = re.sub(r"\(\s*([^()]*?\b(?:plus|minus|mal|durch|hoch)\b[^()]*?)\s*\)", r" Klammer auf \1 Klammer zu ", t)
    t = t.replace("{", " ").replace("}", " ")
    # single letters (variables) by their German names: s -> ess, x -> iks (not inside words or abbreviations)
    t = re.sub(r"(?<![\w.\-'])([A-Za-z])(?![\w\-']|\.\w)", lambda m: LETTERS[m[1].lower()], t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r" ([.,;:!?])", r"\1", t).strip()
