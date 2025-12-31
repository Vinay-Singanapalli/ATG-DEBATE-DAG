from __future__ import annotations

from typing import List, Optional, Dict
import re
import string


_TRANSLATOR = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = t.translate(_TRANSLATOR)
    t = re.sub(r"\s+", " ", t).strip()

    # Normalize some very common stylistic variants so markers match more often
    t = t.replace("long term", "longterm")
    t = t.replace("long-term", "longterm")
    return t


def strip_dynamic_tokens(text: str) -> str:
    t = (text or "")
    t = re.sub(r"\bround\s*\d+\b", " ", t, flags=re.I)
    t = re.sub(r"\bturn\s*\d+\b", " ", t, flags=re.I)
    t = re.sub(r"\b\d+\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_for_repetition(text: str) -> str:
    core = strip_dynamic_tokens(text)
    return normalize_text(core)


def _ngrams(text: str, n: int) -> set[str]:
    t = normalize_text(text)
    if not t:
        return set()
    if len(t) <= n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard_ngram(a: str, b: str, n: int = 4) -> float:
    A = _ngrams(a, n)
    B = _ngrams(b, n)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def near_duplicate_details(
    text: str,
    prior_texts: List[str],
    *,
    ngram_n: int = 4,
    threshold: float = 0.90,
) -> Optional[Dict]:
    best = -1.0
    best_i = -1
    for i, prev in enumerate(prior_texts):
        s = jaccard_ngram(text, prev, n=ngram_n)
        if s > best:
            best = s
            best_i = i

    if best_i >= 0 and best >= threshold:
        return {
            "method": f"jaccard{ngram_n}gram",
            "score": round(best, 4),
            "matched_index": best_i,
            "threshold": threshold,
        }
    return None


_FALLBACK_MARKERS = (
    # Governance / criteria boilerplate
    "staged governance",
    "define a success metric",
    "stop conditions",
    "measurable criteria",
    "decision procedure",
    "limiting principle",
    "legitimacy depends on",

    # Very common repeated openers / templates (topic-agnostic)
    "while the idea of",
    "while the creation of",
    "while the potential benefits",
    "while the technical aspects",
    "may seem appealing",
    "may seem like",
    "it is essential to consider",
    "it is essential to consider the longterm consequences",
    "the debate on",
    "debates about",
    "not only technical but also ethical",
    "not only technical but also moral and political",
    "redistributes risk",
    "redistributes risks",
    "good intentions can still produce harmful governance",

    # Another common template seen in your run (Scientist R5)
    "however when considering",
    "however when considering the potential benefits",
)


def looks_like_fallback(text: str) -> bool:
    t = normalize_text(text)
    return any(m in t for m in _FALLBACK_MARKERS)
