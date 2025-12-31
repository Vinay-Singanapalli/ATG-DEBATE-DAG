from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from nodes.state import DebateState
from nodes.semantic import (
    normalize_for_repetition,
    near_duplicate_details,
    looks_like_fallback,
    normalize_text,
)


def _clean(s: str) -> str:
    return (s or "").strip()


def _extract_line(full: str, label: str) -> str:
    key = label.upper()
    for ln in (full or "").splitlines():
        s = ln.strip()
        if s.upper().startswith(key + " "):
            return s.split(" ", 1)[1].strip()
    return ""


def _first_sentence(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t, maxsplit=1)
    return (parts[0] or "").strip()


def _topic_keywords(topic: str) -> List[str]:
    t = normalize_text(topic)
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "with", "without",
        "using", "use", "is", "are", "be", "should", "could", "would",
    }
    words = [w for w in t.split() if len(w) >= 5 and w not in stop]
    seen = set()
    out: List[str] = []
    for w in words:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out[:8]


def _topic_hit_count(topic: str, text: str) -> int:
    kws = _topic_keywords(topic)
    if not kws:
        return 1
    body = normalize_text(text)
    return sum(1 for k in kws if k in body)


def _possible_contradiction(prev: str, cur: str) -> bool:
    p = normalize_text(prev)
    c = normalize_text(cur)
    neg = ("should not", "must not", "cannot", "never", "no one should")
    pos = ("should", "must", "beneficial", "necessary", "good idea")
    has_neg = any(n in c for n in neg) or any(n in p for n in neg)
    has_pos = any(w in c for w in pos) or any(w in p for w in pos)
    return bool(has_neg and has_pos)


def _parse_pending(pendingtext: str) -> Dict[str, Any]:
    pt = (pendingtext or "").strip()
    if not pt:
        return {"argument": ""}

    if pt.startswith("{") and pt.endswith("}"):
        try:
            data = json.loads(pt)
            if isinstance(data, dict) and isinstance(data.get("argument", ""), str):
                return {"argument": data["argument"]}
        except Exception:
            pass

    rebut = _extract_line(pt, "REBUT")
    new = _extract_line(pt, "NEW")
    arg = " ".join([x for x in [rebut, new] if x]).strip()
    return {"argument": arg}


def _forced_rewrite(topic: str, speaker: str) -> str:
    """
    Topic-safe forced rewrite:
    - Uses the runtime topic string (no hardcoded domain).
    - Avoids common boilerplate openers.
    - 2–4 sentences, single paragraph.
    """
    if speaker == "A":
        return (
            f"On '{topic}', the key is to define measurable success metrics and the specific failure modes that would count as unacceptable harm. "
            f"A staged rollout with independent evaluation can separate optimistic claims from observed outcomes while limiting downside risk. "
            f"If the measured benefits do not exceed costs and harms under realistic conditions, scaling should pause rather than expand."
        )
    return (
        f"Debates about '{topic}' are not only technical but also ethical, because they redistribute risk, power, and responsibility. "
        f"Even if a proposal seems efficient, legitimacy depends on who bears the downside, what rights are protected, and what remedies exist when harm occurs. "
        f"Clear limiting principles prevent the rationale from expanding into unrelated overreach."
    )


def memory_node(state: DebateState) -> DebateState:
    out: Dict[str, Any] = dict(state)
    out["lastnode"] = "MEMORY"
    out["last_node_name"] = "MemoryNode"

    if out.get("status") == "ERROR":
        out["last_node_io"] = {"node": "MEMORY", "input": {}, "output": {"status": "ERROR"}}
        return out

    topic = out.get("topic", "")
    turns: List[Dict[str, Any]] = list(out.get("turns", []))

    speaker = out.get("pendingspeaker")
    agent_name = out.get("pendingagentname")
    pendingtext = (out.get("pendingtext") or "").strip()

    out["last_node_io"] = {
        "node": "MEMORY",
        "input": {
            "pendingspeaker": speaker,
            "pendingagentname": agent_name,
            "pendingtext_preview": pendingtext[:240],
            "turns_len": len(turns),
        },
        "output": {},
    }

    if speaker not in ("A", "B") or not agent_name or not pendingtext:
        out["status"] = "ERROR"
        out["error"] = "MemoryNode missing pending speaker/agent/text."
        out["last_node_io"]["output"] = {"status": "ERROR", "error": out["error"]}
        return out

    max_retries = int(out.get("maxretries", 2))
    retrycount = int(out.get("retrycount", 0))

    parsed = _parse_pending(pendingtext)
    argument = _clean(parsed.get("argument", ""))

    round_no = len(turns) + 1

    # ---------- format checks ----------
    format_issues: List[str] = []
    if len(argument) < 140:
        format_issues.append("argument_too_short")
    if len(argument) > 1100:
        format_issues.append("argument_too_long")
    if "\n" in argument:
        format_issues.append("argument_contains_newlines")

    # ---------- topic drift ----------
    hit_count = _topic_hit_count(topic, argument)
    if hit_count < 1:
        format_issues.append("topic_keywords_missing")
        coherenceflags = list(out.get("coherenceflags", []))
        coherenceflags.append(
            {"round": round_no, "speaker": speaker, "type": "TOPIC_DRIFT_SUSPECTED", "details": {"hit_count": hit_count}}
        )
        out["coherenceflags"] = coherenceflags

    # ---------- repetition (paragraph) ----------
    cand_norm = normalize_for_repetition(argument)
    prior_norm = [normalize_for_repetition(t.get("text", "")) for t in turns]
    dup_any = near_duplicate_details(cand_norm, prior_norm, ngram_n=4, threshold=0.90) if prior_norm else None

    dup_last = None
    if turns:
        last_norm = normalize_for_repetition(turns[-1].get("text", ""))
        dup_last = near_duplicate_details(cand_norm, [last_norm], ngram_n=4, threshold=0.86)

    # ---------- repetition (lead sentence) ----------
    lead = _first_sentence(argument)
    lead_lc = (lead or "").strip().lower()
    if lead_lc.startswith("while "):
        format_issues.append("boilerplate_lead_while")

    lead_norm = normalize_for_repetition(lead)
    prior_leads = [_first_sentence(t.get("text", "")) for t in turns]
    prior_leads_norm = [normalize_for_repetition(x) for x in prior_leads if x]
    dup_lead = near_duplicate_details(lead_norm, prior_leads_norm, ngram_n=4, threshold=0.92) if prior_leads_norm else None

    # ---------- boilerplate / fallback detection ----------
    is_fallback = looks_like_fallback(argument)
    if is_fallback:
        format_issues.append("looks_like_fallback_template")

    reject_reasons: List[str] = []
    if format_issues:
        reject_reasons.extend(format_issues)
    if dup_any is not None:
        reject_reasons.append("duplicate_argument")
    if dup_last is not None:
        reject_reasons.append("duplicate_last_turn")
    if dup_lead is not None:
        reject_reasons.append("duplicate_lead_sentence")

    # ---------- coherence flags (log-only) ----------
    coherenceflags = list(out.get("coherenceflags", []))
    if dup_any or dup_last or dup_lead:
        coherenceflags.append(
            {
                "round": round_no,
                "speaker": speaker,
                "type": "REPETITION_DETECTED",
                "details": {"dup_any": dup_any, "dup_last": dup_last, "dup_lead": dup_lead},
            }
        )

    if turns:
        prev_same_speaker = None
        for t in reversed(turns):
            if t.get("speaker") == speaker:
                prev_same_speaker = t
                break
        if prev_same_speaker and _possible_contradiction(prev_same_speaker.get("text", ""), argument):
            coherenceflags.append(
                {
                    "round": round_no,
                    "speaker": speaker,
                    "type": "POSSIBLE_CONTRADICTION",
                    "details": {"with_round": prev_same_speaker.get("round")},
                }
            )
    out["coherenceflags"] = coherenceflags

    # ---------- rejection / retry ----------
    if reject_reasons:
        rejectionhistory = list(out.get("rejectionhistory", []))
        detail = {
            "reasons": reject_reasons,
            "format_issues": format_issues,
            "dup_any": dup_any,
            "dup_last": dup_last,
            "dup_lead": dup_lead,
            "hit_count": hit_count,
        }

        coherenceflags.append({"round": round_no, "speaker": speaker, "type": "TURN_REJECTED", "details": detail})
        out["coherenceflags"] = coherenceflags
        rejectionhistory.append({"round": round_no, "speaker": speaker, "agent": agent_name, "details": detail})
        out["rejectionhistory"] = rejectionhistory

        # HARD BLOCKS: never accept these after retries.
        hard_block = any(
            r in reject_reasons
            for r in (
                "duplicate_argument",
                "duplicate_last_turn",
                "duplicate_lead_sentence",
                "looks_like_fallback_template",
                "boilerplate_lead_while",
            )
        )

        if retrycount < max_retries:
            out["retrycount"] = retrycount + 1
            out["retryreason"] = ",".join(reject_reasons)[:240]
            out["lastrejectedtext"] = argument

            out["pendingspeaker"] = speaker
            out["pendingagentname"] = agent_name
            out["pendingtext"] = ""

            out["status"] = "OK"
            out["error"] = ""
            out["last_node_io"]["output"] = {"action": "retry", "retrycount": out["retrycount"], "reasons": reject_reasons}
            return out

        if hard_block:
            forced = _forced_rewrite(topic, speaker)
            coherenceflags.append(
                {"round": round_no, "speaker": speaker, "type": "RETRY_EXHAUSTED_FORCED_REWRITE", "details": detail}
            )
            out["coherenceflags"] = coherenceflags
            argument = forced
        else:
            coherenceflags.append(
                {"round": round_no, "speaker": speaker, "type": "RETRY_EXHAUSTED_ACCEPTED", "details": detail}
            )
            out["coherenceflags"] = coherenceflags

    # ACCEPT
    turns.append(
        {
            "round": round_no,
            "agent": agent_name,
            "speaker": speaker,
            "text": argument,
            "meta": {"retrycount": retrycount, "raw_pending": pendingtext[:800]},
        }
    )
    out["turns"] = turns
    out["roundidx"] = len(turns)

    prev_summary = (out.get("summary") or "").strip()
    snippet = argument[:160].strip()
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
    out["last_node_io"]["output"] = {"action": "accept", "round": round_no, "stored_text_preview": argument[:200]}
    return out
