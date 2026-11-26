"""Answer scoring shared by examples/run_basic.py and the analysis scripts,
so a run and its later re-analysis always score answers the same way.
"""

import re


def _normalize(text: str) -> str:
    """Strip formatting noise (thousands separators, curly/straight quote
    variants) that a substring match would otherwise treat as a mismatch.
    """
    text = text.strip().lower()
    text = text.replace(",", "")
    text = text.replace("-", " ")
    text = text.replace("’", "'")
    text = re.sub(r"(\w)'s\b", r"\1", text)  # possessive "'s" -> nothing
    text = text.replace("'", "")             # remaining apostrophes (e.g. "Burrows'")
    return text


def is_correct(model_answer: str, gold_answer: str) -> bool:
    """Naive substring match on normalized text. Good enough for the toy
    dataset — swap for something smarter (embedding similarity, exact-match
    on normalized labels, etc.) once you're using real evaluation data.
    """
    return _normalize(gold_answer) in _normalize(model_answer)


def classify_answer(model_answer: str, classes: list[str]) -> str | None:
    """For fixed-label classification datasets (e.g. verse register), a
    plain substring match is fragile: a short label like "law-wisdom" can
    spuriously match partial answers, or fail to match a differently-worded
    one. Instead, check which of the known class labels appears in the
    normalized answer as a whole word or phrase; return it only if exactly
    one does, else None (ambiguous or no match -> scored wrong).
    """
    normalized_answer = _normalize(model_answer)
    found = [
        c for c in classes
        if re.search(rf"\b{re.escape(_normalize(c))}\b", normalized_answer)
    ]
    return found[0] if len(found) == 1 else None
