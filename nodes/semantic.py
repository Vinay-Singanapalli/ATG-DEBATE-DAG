from __future__ import annotations

from typing import List, Optional, Dict
import re
import string


# -----------------------------
# Vector similarity
# -----------------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


# -----------------------------
# Normalization helpers
# -----------------------------
_TRANSLATOR = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    t = (text or "").lower().strip()
    t = t.translate(_TRANSLATOR)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_core_debate_text(text: str) -> str:
    """
    Remove QUOTE block contents so correct quoting doesn't trigger duplication.
    Keep REBUT/NEW/QUESTION blocks.

    Assumes each block is single-line with prefix "QUOTE:/REBUT:/NEW:/QUESTION:".
    """
    lines = (text or "").splitlines()
    kept: List[str] = []
    in_quote = False

    for line in lines:
        s = line.strip()

        if s.upper().startswith("QUOTE:"):
            in_quote = True
            # Drop quote line entirely (we don't want to dedupe on verbatim quotes)
            continue

        if in_quote and (
            s.upper().startswith("REBUT:")
            or s.upper().startswith("NEW:")
            or s.upper().startswith("QUESTION:")
        ):
            in_quote = False

        if not in_quote:
            kept.append(line)

    return "\n".join(kept).strip()


def strip_dynamic_tokens(text: str) -> str:
    """
    Remove round/speaker counters so templates with changing numbers are still detected as duplicates.
    """
    t = (text or "")
    t = re.sub(r"\bround\s*\d+\b", " ", t, flags=re.I)
    t = re.sub(r"\bspeaker\s*[ab]\b", " ", t, flags=re.I)
    t = re.sub(r"\bturn\s*\d+\b", " ", t, flags=re.I)
    t = re.sub(r"\b\d+\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_for_repetition(text: str) -> str:
    core = extract_core_debate_text(text)
    core = strip_dynamic_tokens(core)
    return normalize_text(core)


# -----------------------------
# Near-duplicate detection (Jaccard on char n-grams)
# -----------------------------
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
    """
    Returns details for the best prior match if similarity >= threshold, else None.
    """
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


# -----------------------------
# Structured turn parsing
# -----------------------------
def extract_block(full_turn: str, block: str) -> str:
    """
    Extract a block value from canonical format: QUOTE/REBUT/NEW/QUESTION.
    """
    key = block.strip().upper() + ":"
    for line in (full_turn or "").splitlines():
        if line.strip().upper().startswith(key):
            return line.split(":", 1)[1].strip()
    return ""


# -----------------------------
# Fallback/template detection
# -----------------------------
# Expanded markers to catch the generic "policy-procedure" fallback content
# that shows up when the model fails validation and your agent returns last_text.
_FALLBACK_MARKERS = (
    # older repeated fallback phrases (your earlier logs)
    "this point misses an important tradeoff",
    "a separate consideration is unintended consequences",
    "what concrete evidence would change your stance",
    # newer generic fallback phrases (from your newer agent/memory behavior)
    "the argument should be evaluated using explicit criteria and evidence",
    "not by repeating earlier wording",
    "propose a concrete decision procedure",
    "measurable criteria and stop conditions",
    "clarify definitions and specify what would falsify your position",
    "what single metric should decide whether the intervention is justified",
    "what stop condition would trigger reversal or redesign",
    "what evidence would you require before scaling",
    "to make progress on",
    "the argument needs concrete criteria and a decision procedure",
)


def looks_like_fallback(text: str) -> bool:
    # Normalize so punctuation/casing doesn't bypass detection
    t = normalize_text(text)
    return any(m in t for m in _FALLBACK_MARKERS)

