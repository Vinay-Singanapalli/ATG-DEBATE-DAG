from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from nodes.state import DebateState


def _clean(s: str) -> str:
    return (s or "").strip()


def _llm_from_state(state: Dict[str, Any], temperature: float) -> ChatOllama:
    model = state.get("llmmodel", "llama3.2:1b")
    max_tokens = int(state.get("llmmaxtokens", 320))
    return ChatOllama(model=model, temperature=temperature, num_predict=max_tokens, format="json")


def _sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if 12 <= len(p) <= 260:
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
    src = _extract_block(opponent_text, "REBUT") or _extract_block(opponent_text, "NEW") or opponent_text
    sents = _sentences(src)
    return sents[0] if sents else ""


def _validate_argument(arg: str) -> Optional[str]:
    arg = _clean(arg)
    if len(arg) < 140:
        return "argument_too_short"
    if len(arg) > 900:
        return "argument_too_long"
    if len(_sentences(arg)) < 2:
        return "argument_needs_multiple_sentences"
    return None


def _agent_turn(state: DebateState, speaker: str) -> DebateState:
    out: Dict[str, Any] = dict(state)

    out["lastnode"] = "AGENT_A" if speaker == "A" else "AGENT_B"
    out["last_node_name"] = "AgentA" if speaker == "A" else "AgentB"

    if out.get("status") == "ERROR":
        out["last_node_io"] = {"node": out["lastnode"], "input": {"status": "ERROR"}, "output": {"status": "ERROR"}}
        return out

    expected = out.get("nextspeaker", "A")
    pending = out.get("pendingspeaker", expected)

    topic = out.get("topic", "")
    agent_name = out.get("agentaname", "Scientist") if speaker == "A" else out.get("agentbname", "Philosopher")
    roundidx = int(out.get("roundidx", 0))

    memory = out.get("memoryfora") if speaker == "A" else out.get("memoryforb")
    last_opp = (memory or {}).get("lastopponentturn") or {}
    opp_text = _clean(last_opp.get("text", ""))

    model_name = out.get("llmmodel", "llama3.2:1b")
    max_tokens = int(out.get("llmmaxtokens", 320))

    retrycount = int(out.get("retrycount", 0))
    retryreason = _clean(out.get("retryreason", ""))
    lastrejected = _clean(out.get("lastrejectedtext", ""))

    out["last_node_io"] = {
        "node": out["lastnode"],
        "input": {
            "speaker": speaker,
            "roundidx": roundidx,
            "expected_nextspeaker": expected,
            "pendingspeaker": pending,
            "agent_name": agent_name,
            "model": model_name,
            "temperature_base": float(out.get("llmtemperature", 0.2)),
            "max_tokens": max_tokens,
            "topic_preview": topic[:120],
            "opp_text_preview": opp_text[:160],
            "retrycount": retrycount,
            "retryreason_preview": retryreason[:120],
        },
        "output": {},
    }

    if expected != speaker or pending != speaker:
        out["status"] = "ERROR"
        out["error"] = (
            "Out-of-turn agent execution. "
            f"expected(nextspeaker)={expected}, pendingspeaker={pending}, called={speaker}"
        )
        out["last_node_io"]["output"] = {"status": "ERROR", "error": out["error"]}
        return out

    persona = (
        "Scientist: argue with mechanisms, real-world failure modes, measurable criteria, and practical safeguards."
        if speaker == "A"
        else "Philosopher: argue with definitions, legitimacy, rights, power, and limiting principles."
    )

    max_retries = int(out.get("maxretries", 2))
    base_temp = float(out.get("llmtemperature", 0.2))
    temps = [base_temp] + [min(0.9, base_temp + 0.15 * i) for i in range(1, max_retries + 1)]

    # If MemoryNode already rejected, don't restart at attempt 0.
    start_i = min(max(retrycount, 0), len(temps) - 1)

    last_raw = ""
    last_reason = ""

    for attempt, temp in enumerate(temps[start_i:], start=start_i):
        system = (
            f"You are {agent_name} in a debate.\n"
            f"Topic: {topic}\n"
            f"Persona: {persona}\n"
            "Return ONLY valid JSON with keys: argument.\n"
            "Hard constraints:\n"
            "- Write one cohesive paragraph of 2–4 sentences.\n"
            "- Stay strictly on the topic.\n"
            "- Do not include headings, bullets, or labels.\n"
            "- Do not ask questions.\n"
            "- Do not start with boilerplate or contrast-openers such as: "
            "'While', 'While the idea', 'While the creation', 'While the technical aspects', 'However,'.\n"
        )

        user = "Write your next round argument."
        if opp_text:
            q = _pick_quote_from_opponent(opp_text)
            if q:
                user += f"\nOpponent last point (respond to it): {q}"

        if retrycount > 0 or attempt > start_i:
            user += "\nThis is a rewrite request."
            if retryreason:
                user += f"\nRejection reason(s): {retryreason}"
            user += "\nDo NOT reuse any full sentence from the rejected draft."
            user += "\nDo NOT begin your first sentence with: While / However / The debate on."
            if lastrejected:
                user += f"\nPrevious rejected text (forbidden to copy): {lastrejected}"

        # record attempt metadata
        out["last_node_io"]["output"] = {
            "attempt": attempt,
            "temperature": temp,
            "start_i": start_i,
            "system_preview": system[:260],
            "user_preview": user[:260],
        }

        llm = _llm_from_state(out, temperature=temp)
        msg = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        raw = getattr(msg, "content", str(msg)).strip()
        last_raw = raw

        try:
            data = json.loads(raw)
        except Exception:
            last_reason = "non_json"
            continue

        argument = _clean(data.get("argument", ""))
        reason = _validate_argument(argument)
        if reason:
            last_reason = reason
            continue

        out["pendingagentname"] = agent_name
        out["pendingtext"] = json.dumps({"argument": argument}, ensure_ascii=False)

        out["last_node_io"]["output"] = {
            "action": "produced_pendingtext",
            "attempt": attempt,
            "temperature": temp,
            "start_i": start_i,
            "argument_preview": argument[:220],
        }
        return out

    # fallback (topic-anchored, not hardcoded)
    if speaker == "A":
        argument = (
            f"'{topic}' should be evaluated by concrete risk-benefit criteria rather than intuition alone. "
            f"A sensible approach is staged deployment with measurable safety targets and clear stop conditions if harms rise. "
            f"That keeps experimentation possible while reducing the chance of irreversible damage."
        )
    else:
        argument = (
            f"Debates about '{topic}' are not only technical but also ethical, because they redistribute risks and power. "
            f"Even if benefits exist, legitimacy depends on consent, accountability, and limiting principles that prevent overreach. "
            f"Without those constraints, good intentions can still produce harmful governance."
        )

    coherenceflags = list(out.get("coherenceflags", []))
    coherenceflags.append(
        {
            "round": roundidx + 1,
            "speaker": speaker,
            "type": "AGENT_FALLBACK_USED",
            "details": {
                "last_reason": last_reason,
                "last_raw_preview": last_raw[:200],
                "start_i": start_i,
            },
        }
    )
    out["coherenceflags"] = coherenceflags

    out["pendingagentname"] = agent_name
    out["pendingtext"] = json.dumps({"argument": argument}, ensure_ascii=False)

    out["last_node_io"]["output"] = {
        "action": "fallback_pendingtext",
        "last_reason": last_reason,
        "last_raw_preview": last_raw[:200],
        "argument_preview": argument[:220],
        "start_i": start_i,
    }
    return out


def agent_a_node(state: DebateState) -> DebateState:
    return _agent_turn(state, "A")


def agent_b_node(state: DebateState) -> DebateState:
    return _agent_turn(state, "B")
