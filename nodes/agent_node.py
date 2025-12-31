from __future__ import annotations
print("LOADED agent_node from:", __file__)
import json
import re
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama

from nodes.state import DebateState


def _clean(s: str) -> str:
    return (s or "").strip()


def _clean_question(q: str) -> str:
    q = (q or "").strip()
    q = q.lstrip("?").strip()
    if q and not q.endswith("?"):
        q += "?"
    return q


def _clean_quote(q: str) -> str:
    q = (q or "").strip()
    while True:
        up = q.upper()
        for lab in ("QUOTE ", "REBUT ", "NEW ", "QUESTION "):
            if up.startswith(lab):
                q = q.split(" ", 1)[1].strip()
                break
        else:
            break
    return q.strip()



def _llm_from_state(state: Dict[str, Any], temperature: float) -> ChatOllama:
    model = state.get("llmmodel", "llama3.2:1b")
    max_tokens = int(state.get("llmmaxtokens", 320))
    # JSON mode still needs validation + retries in practice. [web:117]
    return ChatOllama(model=model, temperature=temperature, num_predict=max_tokens, format="json")


def _sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if 12 <= len(p) <= 240:
            out.append(p)
    return out


def _extract_block(full: str, label: str) -> str:
    key = label.upper()
    for ln in (full or "").splitlines():
        s = ln.strip()
        if s.upper().startswith(key + " "):
            return s.split(" ", 1)[1].strip()
    return ""

def _pick_quote_from_opponent(opponent_text: str) -> str:
    src = _extract_block(opponent_text, "REBUT") or _extract_block(opponent_text, "NEW")
    if not src:
        return "none"
    sents = _sentences(src)
    return sents[0] if sents else "none"



def _validate_fields(roundidx: int, opp_text: str, quote: str, rebut: str, new: str, question: str) -> Optional[str]:
    quote = _clean_quote(quote)
    rebut = _clean(rebut)
    new = _clean(new)
    question = _clean_question(question)

    if len(rebut) < 60:
        return "rebut_too_short"
    if len(new) < 60:
        return "new_too_short"
    if not question or not question.endswith("?") or len(question) < 12:
        return "question_invalid"

    if roundidx == 0:
        if quote.lower() != "none":
            return "quote_must_be_none_round1"
    else:
        if quote.lower() == "none":
            return "quote_cannot_be_none_after_round1"
        if not opp_text or quote not in opp_text:
            return "quote_not_from_opponent"

    # discourage NEW being a copy of REBUT
    if re.sub(r"\W+", "", rebut.lower()) == re.sub(r"\W+", "", new.lower()):
        return "new_duplicates_rebut"

    return None


def _topic_anchored_fallback(topic: str, speaker: str, roundidx: int, opp_text: str) -> Dict[str, str]:
    # Generic across domains; no hardcoded topic assumptions.
    quote = "none" if roundidx == 0 else _clean_quote(_pick_quote_from_opponent(opp_text))


    if speaker == "A":
        rebut = (
            f"The quote is relevant, but it does not yet establish the key causal mechanism for '{topic}'. "
            f"A stronger case should specify what outcome is being optimized, what evidence would change the conclusion, "
            f"and what measurable indicators would show success or failure."
        )
        new = (
            f"A pragmatic way to evaluate '{topic}' is staged governance: define a success metric, run limited pilots, "
            f"measure benefits and harms against a baseline, and adopt explicit stop conditions if risks exceed bounds."
        )
        question = f"What single measurable outcome would most strongly support your position on '{topic}' within one year?"
    else:
        rebut = (
            f"The quote emphasizes outcomes, but '{topic}' also requires a legitimacy check: who bears the risks, "
            f"who benefits, and what constraints should limit pursuit of the goal. Without a limiting principle, the argument can overreach."
        )
        new = (
            f"An ethical analysis of '{topic}' should separate moral status (who counts), protections (what is owed), "
            f"and accountability (who decides and what remedy exists if harm occurs). This prevents hidden value assumptions."
        )
        question = f"Which right or constraint should never be overridden when pursuing '{topic}', and why?"

    return {"quote": quote, "rebut": rebut, "new": new, "question": question}


def _agent_turn(state: DebateState, speaker: str) -> DebateState:
    out: Dict[str, Any] = dict(state)
    out["lastnode"] = "AGENT_A" if speaker == "A" else "AGENT_B"

    if out.get("status") == "ERROR":
        return out

    expected = out.get("nextspeaker", "A")
    if not out.get("pendingspeaker"):
        out["pendingspeaker"] = expected
    pending = out.get("pendingspeaker")

    if expected != speaker or pending != speaker:
        out["status"] = "ERROR"
        out["error"] = (
            "Out-of-turn agent execution. "
            f"expected(nextspeaker)={expected}, pendingspeaker={pending}, called={speaker}"
        )
        return out

    topic = out.get("topic", "")
    agent_name = out.get("agentaname", "Scientist") if speaker == "A" else out.get("agentbname", "Philosopher")
    roundidx = int(out.get("roundidx", 0))

    memory = out.get("memoryfora") if speaker == "A" else out.get("memoryforb")
    last_opp = (memory or {}).get("lastopponentturn") or {}
    opp_text = _clean(last_opp.get("text", ""))

    persona = (
        "Scientist: use mechanisms, measurable criteria, uncertainty, and testable claims."
        if speaker == "A"
        else "Philosopher: use definitions, legitimacy, ethical constraints, and limiting principles."
    )

    max_retries = int(out.get("maxretries", 2))
    base_temp = float(out.get("llmtemperature", 0.2))
    temps = [base_temp] + [min(0.9, base_temp + 0.15 * i) for i in range(1, max_retries + 1)]

    retrycount = int(out.get("retrycount", 0))
    retryreason = _clean(out.get("retryreason", ""))
    lastrejected = _clean(out.get("lastrejectedtext", ""))

    last_raw = ""
    last_reason = ""

    for attempt, temp in enumerate(temps, start=0):
        system = (
            f"You are {agent_name} in a structured debate.\n"
            f"Topic: {topic}\n"
            f"Persona: {persona}\n"
            "Return ONLY valid JSON with keys: quote, rebut, new, question.\n"
            "Hard constraints:\n"
            "- Stay strictly on the Topic.\n"
            "- Round 1: quote must be exactly 'none'.\n"
            "- Later rounds: quote must be exactly ONE sentence copied verbatim from the opponent turn provided.\n"
            "- The quote field must NOT include a leading 'QUOTE' label.\n"
            "- rebut responds to quote directly.\n"
            "- new adds a distinct argument.\n"
            "- question is one pointed question ending with '?'.\n"
            "- rebut and new must each be at least 60 characters.\n"
            "- No markdown, no extra keys.\n"
        )

        user = "Write your next turn."
        if opp_text:
            user += "\nOpponent last turn (quote exactly ONE sentence from this when roundidx>0):\n" + opp_text

        if retrycount > 0 or attempt > 0:
            user += "\nThis is a rewrite request."
            if retryreason:
                user += f"\nRejection reason(s): {retryreason}"
            if lastrejected:
                user += "\nPrevious rejected turn (do NOT reuse sentences from it):\n" + lastrejected
            user += f"\nReminder: stay on topic: {topic}"

        llm = _llm_from_state(out, temperature=temp)
        msg = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        raw = getattr(msg, "content", str(msg)).strip()
        last_raw = raw

        try:
            data = json.loads(raw)
        except Exception:
            last_reason = "non_json"
            continue

        quote = _clean_quote(data.get("quote", ""))
        rebut = _clean(data.get("rebut", ""))
        new = _clean(data.get("new", ""))
        question = _clean_question(data.get("question", ""))

        if roundidx == 0:
            quote = "none"

        reason = _validate_fields(roundidx, opp_text, quote, rebut, new, question)
        if reason:
            last_reason = reason
            continue

        out["pendingagentname"] = agent_name
        out["pendingtext"] = f"QUOTE {quote}\nREBUT {rebut}\nNEW {new}\nQUESTION {question}"
        return out

    fb = _topic_anchored_fallback(topic, speaker, roundidx, opp_text)

    coherenceflags = list(out.get("coherenceflags", []))
    coherenceflags.append(
        {
            "round": roundidx + 1,
            "speaker": speaker,
            "type": "AGENT_FALLBACK_USED",
            "details": {"last_reason": last_reason, "last_raw_preview": last_raw[:200]},
        }
    )
    out["coherenceflags"] = coherenceflags

    out["pendingagentname"] = agent_name
    out["pendingtext"] = f"QUOTE {fb['quote']}\nREBUT {fb['rebut']}\nNEW {fb['new']}\nQUESTION {fb['question']}"
    return out


def agent_a_node(state: DebateState) -> DebateState:
    return _agent_turn(state, "A")


def agent_b_node(state: DebateState) -> DebateState:
    return _agent_turn(state, "B")

