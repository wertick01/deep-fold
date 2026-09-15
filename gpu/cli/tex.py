"""LaTeX math → Unicode for ``deepfold chat``. No extra dependency.

The terminal cannot paint KaTeX. This is a readable approximation of the
subset instruct models actually emit: greek, operators, ``\\frac``,
``\\sqrt``, sub/superscripts, and small matrices.
"""

from __future__ import annotations

__all__ = ["latex_to_unicode"]

_GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}

_SYMS = {
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "cdot": "·",
    "ast": "*",
    "star": "⋆",
    "div": "÷",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "equiv": "≡",
    "sim": "∼",
    "simeq": "≃",
    "cong": "≅",
    "propto": "∝",
    "infty": "∞",
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "oint": "∮",
    "partial": "∂",
    "nabla": "∇",
    "emptyset": "∅",
    "varnothing": "∅",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "neg": "¬",
    "lnot": "¬",
    "in": "∈",
    "notin": "∉",
    "ni": "∋",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "cup": "∪",
    "cap": "∩",
    "vee": "∨",
    "wedge": "∧",
    "oplus": "⊕",
    "otimes": "⊗",
    "circ": "∘",
    "bullet": "•",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "leftrightarrow": "↔",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "mapsto": "↦",
    "uparrow": "↑",
    "downarrow": "↓",
    "ldots": "…",
    "cdots": "⋯",
    "vdots": "⋮",
    "ddots": "⋱",
    "hbar": "ℏ",
    "ell": "ℓ",
    "Re": "ℜ",
    "Im": "ℑ",
    "wp": "℘",
    "angle": "∠",
    "perp": "⊥",
    "parallel": "∥",
    "langle": "⟨",
    "rangle": "⟩",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "vert": "|",
    "mid": "∣",
    "backslash": "\\",
    "dagger": "†",
    "ddagger": "‡",
    "prime": "′",
    "degree": "°",
    "percent": "%",
    "dots": "…",
}

_SKIP = frozenset(
    {
        "displaystyle",
        "textstyle",
        "scriptstyle",
        "limits",
        "nolimits",
        "mathord",
        "mathrel",
        "mathbin",
        "mathop",
        "left",
        "right",
        "big",
        "Big",
        "bigg",
        "Bigg",
        "bigl",
        "bigr",
        "Bigl",
        "Bigr",
        "quad",
        "qquad",
        "hspace",
        "vspace",
        "qquad",
        "!",
    }
)

_UNWRAP = frozenset(
    {
        "mathrm",
        "mathbf",
        "mathit",
        "mathsf",
        "mathtt",
        "mathcal",
        "mathscr",
        "mathfrak",
        "operatorname",
        "text",
        "textbf",
        "textit",
        "textrm",
        "mbox",
        "hbox",
        "overline",
        "underline",
        "hat",
        "bar",
        "tilde",
        "vec",
        "dot",
        "ddot",
        "widehat",
        "widetilde",
        "boldsymbol",
        "bm",
    }
)

_ACCENT = {
    "hat": "\u0302",
    "tilde": "\u0303",
    "bar": "\u0304",
    "overline": "\u0305",
    "vec": "\u20d7",
    "dot": "\u0307",
    "ddot": "\u0308",
    "widehat": "\u0302",
    "widetilde": "\u0303",
}

_BB = {
    "C": "ℂ",
    "H": "ℍ",
    "N": "ℕ",
    "P": "ℙ",
    "Q": "ℚ",
    "R": "ℝ",
    "Z": "ℤ",
}

_FUNS = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "log",
        "ln",
        "exp",
        "max",
        "min",
        "sup",
        "inf",
        "lim",
        "det",
        "dim",
        "ker",
        "arg",
        "sinh",
        "cosh",
        "tanh",
        "arcsin",
        "arccos",
        "arctan",
        "Pr",
        "tr",
        "diag",
        "rank",
        "sgn",
    }
)

_FRACTIONS = {
    ("1", "2"): "½",
    ("1", "3"): "⅓",
    ("2", "3"): "⅔",
    ("1", "4"): "¼",
    ("3", "4"): "¾",
    ("1", "5"): "⅕",
    ("2", "5"): "⅖",
    ("3", "5"): "⅗",
    ("4", "5"): "⅘",
    ("1", "6"): "⅙",
    ("5", "6"): "⅚",
    ("1", "8"): "⅛",
    ("3", "8"): "⅜",
    ("5", "8"): "⅝",
    ("7", "8"): "⅞",
}

_SUPER = str.maketrans(
    "0123456789+-=()n",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ",
)
_SUPER_EXTRA = {
    "i": "ⁱ",
    "T": "ᵀ",
    "a": "ᵃ",
    "e": "ᵉ",
    "o": "ᵒ",
    "x": "ˣ",
    "k": "ᵏ",
    "m": "ᵐ",
    "p": "ᵖ",
    "t": "ᵗ",
    "h": "ʰ",
    "r": "ʳ",
    "s": "ˢ",
    "v": "ᵛ",
    "j": "ʲ",
}
_SUB = str.maketrans(
    "0123456789+-=()aeoxhklmnpst",
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ",
)
_SUB_EXTRA = {"i": "ᵢ", "j": "ⱼ", "r": "ᵣ", "u": "ᵤ", "v": "ᵥ"}


def latex_to_unicode(src: str) -> str:
    """Best-effort. Never raises on junk the model emitted."""
    text = (src or "").strip()
    if not text:
        return ""
    try:
        return _squeeze(_parse(text))
    except Exception:
        return text


def _squeeze(text: str) -> str:
    out: list[str] = []
    prev_space = False
    for ch in text:
        if ch in "\t\r":
            ch = " "
        if ch == " ":
            if prev_space:
                continue
            prev_space = True
            out.append(ch)
            continue
        prev_space = False
        out.append(ch)
    return "".join(out).strip()


def _parse(s: str) -> str:
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            piece, i = _command(s, i)
            out.append(piece)
            continue
        if ch == "{":
            inner, i = _group(s, i)
            out.append(_parse(inner))
            continue
        if ch == "}":
            i += 1
            continue
        if ch == "^":
            arg, i = _raw_atom(s, i + 1)
            out.append(_script(_parse(arg), "sup"))
            continue
        if ch == "_":
            arg, i = _raw_atom(s, i + 1)
            out.append(_script(_parse(arg), "sub"))
            continue
        if ch == "&":
            out.append("  ")
            i += 1
            continue
        if ch in "\n\t":
            if not out or out[-1] != " ":
                out.append(" ")
            i += 1
            continue
        if ch == "~":
            out.append(" ")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _read_cmd(s: str, i: int) -> tuple[str, int]:
    i += 1
    if i >= len(s):
        return "", i
    ch = s[i]
    if ch.isalpha():
        j = i
        while j < len(s) and s[j].isalpha():
            j += 1
        return s[i:j], j
    return ch, i + 1


def _group(s: str, i: int) -> tuple[str, int]:
    if i >= len(s) or s[i] != "{":
        return "", i
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "\\" and j + 1 < len(s) and s[j + 1] in "{}\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
        j += 1
    return s[i + 1 :], len(s)


def _raw_atom(s: str, i: int) -> tuple[str, int]:
    while i < len(s) and s[i] in " \t":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return _group(s, i)
    if s[i] == "\\":
        name, j = _read_cmd(s, i)
        return "\\" + name, j
    return s[i], i + 1


def _take(s: str, i: int) -> tuple[str, int]:
    raw, j = _raw_atom(s, i)
    return _parse(raw), j


def _command(s: str, i: int) -> tuple[str, int]:
    name, i = _read_cmd(s, i)
    if not name:
        return "\\", i
    if name == "\\":
        return "\n", i
    if name in {",", ":", ";", " "}:
        return " ", i
    if name == "!":
        return "", i
    if name in {"%", "&", "#", "_", "$", "{", "}"}:
        return name, i
    if name in _SKIP:
        if name in {"hspace", "vspace"} and i < len(s) and s[i] == "{":
            _, i = _group(s, i)
        return "", i
    if name in ("frac", "dfrac", "tfrac", "cfrac", "binom"):
        num, i = _take(s, i)
        den, i = _take(s, i)
        if name == "binom":
            return f"C({num},{den})", i
        return _frac(num, den), i
    if name == "sqrt":
        index = ""
        if i < len(s) and s[i] == "[":
            close = s.find("]", i + 1)
            if close >= 0:
                index = _parse(s[i + 1 : close])
                i = close + 1
        body, i = _take(s, i)
        if index:
            return f"{_script(index, 'sup')}√({body})", i
        if len(body) == 1:
            return f"√{body}", i
        return f"√({body})", i
    if name == "mathbb":
        body, i = _take(s, i)
        return "".join(_BB.get(ch, ch) for ch in body), i
    if name in _ACCENT:
        body, i = _take(s, i)
        if not body:
            return "", i
        comb = _ACCENT[name]
        return body[0] + comb + body[1:], i
    if name in _UNWRAP:
        body, i = _take(s, i)
        return body, i
    if name == "not":
        nxt, i = _take(s, i) if i < len(s) and s[i] == "\\" else ("", i)
        if nxt.startswith("∈") or nxt == "∈":
            return "∉", i
        if nxt == "=":
            return "≠", i
        return "¬" + nxt, i
    if name == "begin":
        env, i = _raw_atom(s, i)
        env = env.strip()
        tag = "\\end{" + env + "}"
        k = s.find(tag, i)
        if k < 0:
            body, i = s[i:], len(s)
        else:
            body, i = s[i:k], k + len(tag)
        return _env(env, body), i
    if name == "end":
        if i < len(s) and s[i] == "{":
            _, i = _group(s, i)
        return "", i
    if name in _GREEK:
        return _GREEK[name], i
    if name in _SYMS:
        return _SYMS[name], i
    if name in _FUNS:
        return name, i
    return name, i


def _frac(num: str, den: str) -> str:
    key = (num, den)
    if key in _FRACTIONS:
        return _FRACTIONS[key]
    return f"{_paren(num)}/{_paren(den)}"


def _paren(text: str) -> str:
    if not text:
        return text
    if text.startswith("(") and text.endswith(")") and text.count("(") == text.count(")"):
        return text
    if any(ch in text for ch in "+-=<>/ "):
        return f"({text})"
    return text


def _script(text: str, kind: str) -> str:
    if not text:
        return ""
    extra = _SUPER_EXTRA if kind == "sup" else _SUB_EXTRA
    trans = _SUPER if kind == "sup" else _SUB
    wrap = "^" if kind == "sup" else "_"
    out: list[str] = []
    for ch in text:
        if ch == " ":
            continue
        if ch in extra:
            out.append(extra[ch])
            continue
        converted = ch.translate(trans)
        if converted != ch:
            out.append(converted)
            continue
        return wrap + _paren(text)
    return "".join(out)


def _env(env: str, body: str) -> str:
    env = env.strip()
    rows = _split_top(body, "\\\\")
    grid = [[_parse(cell.strip()) for cell in _split_top(row, "&")] for row in rows]
    grid = [row for row in grid if any(cell for cell in row)]
    if not grid:
        return _parse(body)
    if env in {"align", "aligned", "gather", "gathered", "eqnarray", "cases"}:
        return "\n".join("  ".join(row) for row in grid)
    left, right = "(", ")"
    if env in {"bmatrix", "Bmatrix"}:
        left, right = "[", "]"
    elif env == "vmatrix":
        left, right = "|", "|"
    elif env == "Vmatrix":
        left, right = "∥", "∥"
    elif env not in {"matrix", "pmatrix", "smallmatrix"}:
        return "\n".join("  ".join(row) for row in grid)
    widths = [0] * max(len(row) for row in grid)
    for row in grid:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines: list[str] = []
    for row in grid:
        padded = [
            row[idx].ljust(widths[idx]) if idx < len(row) else " " * widths[idx]
            for idx in range(len(widths))
        ]
        lines.append(f"{left} {'  '.join(padded)} {right}")
    return "\n".join(lines)


def _split_top(s: str, sep: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    depth = 0
    n = len(s)
    while i < n:
        if s[i] == "\\" and i + 1 < n and s[i + 1] in "{}":
            buf.append(s[i : i + 2])
            i += 2
            continue
        if s[i] == "{":
            depth += 1
            buf.append(s[i])
            i += 1
            continue
        if s[i] == "}":
            depth = max(0, depth - 1)
            buf.append(s[i])
            i += 1
            continue
        if depth == 0 and s.startswith(sep, i):
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(s[i])
        i += 1
    parts.append("".join(buf))
    return parts
