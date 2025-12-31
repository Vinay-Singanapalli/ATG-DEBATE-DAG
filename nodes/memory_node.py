from __future__ import annotations
print("LOADED memory_node from:", __file__)
import re
from typing import Any, Dict, List, Optional, Set

from nodes.state import DebateState

def _forced_rewrite(topic: str, speaker: str, round_no: int) -> str:
    # Guaranteed non-empty, on-topic, and structurally valid.
    if speaker == "A":
        rebut = (
            f"For '{topic}', the key question is whether expected benefits justify the opportunity cost and the specific risks. "
            f"Policy should be evaluated against a clear baseline, not against vague optimism or fear."
        )
        new = (
            f"A practical safeguard is staged funding with public milestones: publish objectives, measure outputs, "
            f"and stop or redesign the program if targets are missed. This keeps '{topic}' accountable."
        )
        question = f"What would be your clearest stop-condition for '{topic}' if early results look negative?"
    else:
        rebut = (
            f"Even if '{topic}' has benefits, legitimacy depends on who bears harms and who decides. "
            f"If impacts fall on a minority, consent, compensation, and enforceable limits are required."
        )
        new = (
            f"A limiting principle is essential: specify exactly when the reasoning for '{topic}' applies and when it does not, "
            f"so the policy cannot expand into unrelated overreach."
        )
        question = f"Which group bears the main risk from '{topic}', and what concrete remedy do they get if harmed?"

    return f"QUOTE none\nREBUT {rebut}\nNEW {new}\nQUESTION {question}"



def _extract_line(full: str, label: str) -> str:
    key = label.upper()
    for ln in (full or "").splitlines():
        s = ln.strip()
        if s.upper().startswith(key + " "):
            return s.split(" ", 1)[1].strip()
    return ""


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_question(q: str) -> str:
    q = (q or "").strip()
    q = q.lstrip("?").strip()
    if q and not q.endswith("?"):
        q = q + "?"
    return q


def _topic_keywords(topic: str) -> List[str]:
    t = _normalize_text(topic)
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "without",
        "using", "use", "is", "are", "be", "should", "could", "would", "giving",
    }
    words = [w for w in t.split() if len(w) >= 5 and w not in stop]
    seen = set()
    out: List[str] = []
    for w in words:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out[:8]


def _topic_hit_count(topic: str, rebut: str, new: str) -> int:
    kws = _topic_keywords(topic)
    if not kws:
        return 1
    body = _normalize_text(rebut + " " + new)
    return sum(1 for k in kws if k in body)


def _core_text(rebut: str, new: str, question: str) -> str:
    return _normalize_text(f"REBUT {rebut}\nNEW {new}\nQUESTION {question}")


def _ngrams(s: str, n: int = 4) -> Set[str]:
    t = _normalize_text(s)
    if not t:
        return set()
    if len(t) <= n:
        return {t}
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def _jaccard(a: str, b: str, n: int = 4) -> float:
    A = _ngrams(a, n=n)
    B = _ngrams(b, n=n)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _best_duplicate(candidate: str, prior_texts: List[str], threshold: float, n: int = 4) -> Optional[Dict[str, Any]]:
    best = -1.0
    best_i = -1
    for i, prev in enumerate(prior_texts):
        s = _jaccard(candidate, prev, n=n)
        if s > best:
            best = s
            best_i = i
    if best_i >= 0 and best >= threshold:
        return {"method": f"jaccard_{n}gram", "score": round(best, 4), "matched_index": best_i, "threshold": threshold}
    return None


def memory_node(state: DebateState) -> DebateState:
    print("RUNNING memory_node version: 2025-12-31-22:15")
    out: Dict[str, Any] = dict(state)
    out["lastnode"] = "MEMORY"

    if out.get("status") == "ERROR":
        return out

    topic = out.get("topic", "")
    turns: List[Dict[str, Any]] = list(out.get("turns", []))

    speaker = out.get("pendingspeaker")
    agent_name = out.get("pendingagentname")
    raw_text = (out.get("pendingtext") or "").strip()

    if speaker not in ("A", "B") or not agent_name or not raw_text:
        out["status"] = "ERROR"
        out["error"] = "MemoryNode missing pending speaker/agent/text."
        return out

    max_retries = int(out.get("maxretries", 2))
    retrycount = int(out.get("retrycount", 0))

    # Extract fields
    quote = _extract_line(raw_text, "QUOTE").strip()
    rebut = _extract_line(raw_text, "REBUT").strip()
    new = _extract_line(raw_text, "NEW").strip()
    question = _clean_question(_extract_line(raw_text, "QUESTION"))

    # Canonical storage text
    text = f"QUOTE {quote}\nREBUT {rebut}\nNEW {new}\nQUESTION {question}"

    round_no = len(turns) + 1

    # IMPORTANT: define this BEFORE any appends
    format_issues: List[str] = []

    # Basic structure checks
    if not quote:
        format_issues.append("missing_quote")
    if not rebut or len(rebut) < 60:
        format_issues.append("rebut_too_short")
    if not new or len(new) < 60:
        format_issues.append("new_too_short")
    if not question or question in ("?", "??") or len(question) < 12:
        format_issues.append("question_invalid")
    if not question.endswith("?"):
        format_issues.append("question_missing_qmark")

    # Reject quotes that contain labels (prevents QUOTE QUOTE..., QUOTE QUESTION..., etc.)
    bad_prefixes = ("QUOTE ", "REBUT ", "NEW ", "QUESTION ")
    if quote.upper().startswith(bad_prefixes):
        format_issues.append("quote_contains_label")

    # Quote rule enforcement
    if round_no == 1:
        if quote.lower() != "none":
            format_issues.append("quote_must_be_none_round1")
    else:
        if quote.lower() == "none":
            format_issues.append("quote_cannot_be_none_after_round1")

        opp_speaker = "B" if speaker == "A" else "A"
        opp_last = None
        for t in reversed(turns):
            if t.get("speaker") == opp_speaker:
                opp_last = t
                break
        opp_text = (opp_last or {}).get("text", "")
        if not opp_text or quote not in opp_text:
            format_issues.append("quote_not_from_opponent")

    # Topic gate
    if _topic_hit_count(topic, rebut, new) < 1:
        format_issues.append("topic_keywords_missing")

    # No-repeat checks (core only)
    candidate_core = _core_text(rebut, new, question)

    prior_same: List[str] = []
    prior_all: List[str] = []
    for t in turns:
        t_text = t.get("text", "")
        t_rebut = _extract_line(t_text, "REBUT")
        t_new = _extract_line(t_text, "NEW")
        t_q = _clean_question(_extract_line(t_text, "QUESTION"))
        t_core = _core_text(t_rebut, t_new, t_q)

        prior_all.append(t_core)
        if t.get("speaker") == speaker:
            prior_same.append(t_core)

    dup_same = _best_duplicate(candidate_core, prior_same, threshold=0.86, n=4)
    dup_any = _best_duplicate(candidate_core, prior_all, threshold=0.90, n=4)

    # Prevent immediate mirror-copying of the last turn
    dup_last = None
    if turns:
        last = turns[-1].get("text", "")
        last_rebut = _extract_line(last, "REBUT")
        last_new = _extract_line(last, "NEW")
        last_q = _clean_question(_extract_line(last, "QUESTION"))
        last_core = _core_text(last_rebut, last_new, last_q)
        dup_last = _best_duplicate(candidate_core, [last_core], threshold=0.84, n=4)

    reject_reasons: List[str] = []
    if format_issues:
        reject_reasons.extend(format_issues)
    if dup_same is not None:
        reject_reasons.append("duplicate_same_speaker")
    if dup_any is not None:
        reject_reasons.append("duplicate_cross_speaker")
    if dup_last is not None:
        reject_reasons.append("duplicate_last_turn")

    if reject_reasons:
        coherenceflags = list(out.get("coherenceflags", []))
        rejectionhistory = list(out.get("rejectionhistory", []))

        detail = {
            "reasons": reject_reasons,
            "format_issues": format_issues,
            "dup_same": dup_same,
            "dup_any": dup_any,
            "dup_last": dup_last,
        }

        # Retries exhausted: DO NOT accept the repeated/bad content.
        # # Force a local rewrite so we still reach 8 rounds without violating constraints.
        coherenceflags.append({"round": round_no, "speaker": speaker, "type": "RETRY_EXHAUSTED_FORCED_REWRITE", "details": detail})
        out["coherenceflags"] = coherenceflags
        text = _forced_rewrite(topic, speaker, round_no)

        if round_no > 1:
            opp_speaker = "B" if speaker == "A" else "A"
            opp_last = None
            for t in reversed(turns):
                if t.get("speaker") == opp_speaker:
                    opp_last = t
                    break
            opp_text = (opp_last or {}).get("text", "")
            # Quote opponent REBUT if possible
            qsrc = _extract_line(opp_text, "REBUT") or _extract_line(opp_text, "NEW") or opp_text
            qsent = qsrc.split(".")[0].strip()
            if qsent and len(qsent) > 12:
                text = text.replace("QUOTE none", f"QUOTE {qsent}", 1)

        # re-extract for consistent memory slices
        quote = _extract_line(text, "QUOTE").strip()
        rebut = _extract_line(text, "REBUT").strip()
        new = _extract_line(text, "NEW").strip()
        question = _clean_question(_extract_line(text, "QUESTION"))


        if retrycount < max_retries:
            out["retrycount"] = retrycount + 1
            out["retryreason"] = ",".join(reject_reasons)[:240]
            out["lastrejectedtext"] = text

            out["pendingspeaker"] = speaker
            out["pendingagentname"] = agent_name
            out["pendingtext"] = ""

            out["status"] = "OK"
            out["error"] = ""
            return out

        coherenceflags.append({"round": round_no, "speaker": speaker, "type": "RETRY_EXHAUSTED_ACCEPTED", "details": detail})
        out["coherenceflags"] = coherenceflags

    # ACCEPT
    turns.append({"round": round_no, "agent": agent_name, "speaker": speaker, "text": text, "meta": {"retrycount": retrycount}})
    out["turns"] = turns
    out["roundidx"] = len(turns)

    prev_summary = (out.get("summary") or "").strip()
    snippet = text.replace("\n", " ")[:160].strip()
    if not prev_summary:
        out["summary"] = f"Topic: {topic}. R{round_no} {agent_name}: {snippet}"
    else:
        out["summary"] = (prev_summary + f" | R{round_no} {agent_name}: {snippet}")[-900:]

    out["nextspeaker"] = "B" if speaker == "A" else "A"

    def last_turn_for(s: str) -> Optional[Dict[str, Any]]:
        for t in reversed(turns):
            if t.get("speaker") == s:
                return t
        return None

    a_last = last_turn_for("A")
    b_last = last_turn_for("B")
    recent = [{"round": t["round"], "agent": t["agent"], "text": t["text"]} for t in turns[-3:]]

    out["memoryfora"] = {
        "summary": out["summary"][-700:],
        "recentturns": recent,
        "lastownturn": {"round": a_last["round"], "text": a_last["text"]} if a_last else None,
        "lastopponentturn": {"round": b_last["round"], "text": b_last["text"]} if b_last else None,
        "youare": "AgentA",
    }
    out["memoryforb"] = {
        "summary": out["summary"][-700:],
        "recentturns": recent,
        "lastownturn": {"round": b_last["round"], "text": b_last["text"]} if b_last else None,
        "lastopponentturn": {"round": a_last["round"], "text": a_last["text"]} if a_last else None,
        "youare": "AgentB",
    }

    out["retrycount"] = 0
    out["retryreason"] = ""
    out["lastrejectedtext"] = ""

    out["pendingspeaker"] = out["nextspeaker"]
    out["pendingagentname"] = ""
    out["pendingtext"] = ""

    out["status"] = "OK"
    out["error"] = ""
    return out
