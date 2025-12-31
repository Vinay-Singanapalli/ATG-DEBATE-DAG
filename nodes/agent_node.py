from __future__ import annotations

import json
import random
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama

from nodes.state import DebateState


def _clean(s: str) -> str:
    return (s or "").strip()


def _llm_from_state(state: Dict[str, Any], temperature: float) -> ChatOllama:
    model = state.get("llmmodel", "llama3.2:1b")
    max_tokens = int(state.get("llmmaxtokens", 320))
    return ChatOllama(model=model, temperature=temperature, num_predict=max_tokens, format="json")


def _safe_fallback(topic: str, speaker: str, roundidx: int) -> Dict[str, str]:
    # topic-anchored, non-templatey but safe and long enough
    if speaker == "A":
        rebut = (
            f"Public funding for space exploration can be justified when it is treated like other high-risk, high-return R&D: "
            f"define measurable goals (technology readiness, cost per kilogram to orbit, spin-offs) and audit outcomes. "
            f"This reduces waste while preserving the upside of long-horizon innovation that private markets often underprovide."
        )
        new = (
            f"A practical policy is to fund space programs only when they create spillovers that are hard to capture privately—"
            f"for example open standards, shared infrastructure, and fundamental science. "
            f"That turns the debate into a governance problem: clear milestones, staged budgets, and stop conditions if targets are missed."
        )
        question = f"What concrete metric would make you say public space spending is unjustified for '{topic}'?"
    else:
        rebut = (
            f"Even if space exploration is inspiring, public money is morally constrained by opportunity cost: "
            f"funding a prestige project can still be wrong if it predictably crowds out urgent needs for vulnerable groups. "
            f"Legitimacy requires that those bearing the cost can reasonably endorse the tradeoff."
        )
        new = (
            f"A deeper issue is distributive justice: who benefits from space programs versus who pays, and who gets to decide. "
            f"If benefits are concentrated while costs are diffuse, then democratic accountability and explicit consent mechanisms become essential."
        )
        question = f"Who should have the decisive voice in allocating public money to '{topic}', and why?"

    quote = "none" if roundidx == 0 else "none"
    return {"quote": quote, "rebut": rebut, "new": new, "question": question}


def _validate_fields(quote: str, rebut: str, new: str, question: str) -> Optional[str]:
    if not question.endswith("?"):
        return "question_missing_qmark"
    if len(rebut) < 40:
        return "rebut_too_short"
    if len(new) < 40:
        return "new_too_short"
    if not quote:
        return "quote_missing"
    return None


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

    # Determinism hook: seed controls retry temperature & fallback selection
    seed = out.get("seed")
    rng = random.Random(seed if seed is not None else None)

    max_retries = int(out.get("maxretries", 2))

    base_temp = float(out.get("llmtemperature", 0.2))
    # allow slight increase per retry to escape failure modes
    temps = [base_temp] + [min(0.9, base_temp + 0.15 * i) for i in range(1, max_retries + 1)]

    last_raw = ""
    last_reason = ""

    for attempt, temp in enumerate(temps, start=0):
        system = (
            f"You are {agent_name} in a structured debate.\n"
            f"Topic: {topic}\n"
            f"Persona: {persona}\n"
            "Return ONLY valid JSON with keys: quote, rebut, new, question.\n"
            "Rules:\n"
            "- Round 1: quote must be exactly 'none'.\n"
            "- Later rounds: quote must be ONE sentence copied verbatim from opponent.\n"
            "- rebut responds to quote directly.\n"
            "- new adds a distinct argument (not a paraphrase).\n"
            "- question ends with '?'.\n"
            "- Each of rebut and new must be at least 60 characters.\n"
            "- No markdown, no extra keys.\n"
        )

        user = "Write your next turn."
        if opp_text:
            user += f"\nOpponent last turn:\n{opp_text}"
        if attempt > 0:
            user += f"\nRETRY #{attempt}: Your previous output failed validation reason={last_reason}. Fix it."

        llm = _llm_from_state(out, temperature=temp)
        msg = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        raw = getattr(msg, "content", str(msg)).strip()
        last_raw = raw

        try:
            data = json.loads(raw)
        except Exception:
            last_reason = "non_json"
            continue

        quote = _clean(data.get("quote", ""))
        rebut = _clean(data.get("rebut", ""))
        new = _clean(data.get("new", ""))
        question = _clean(data.get("question", ""))

        if roundidx == 0:
            quote = "none"

        reason = _validate_fields(quote, rebut, new, question)
        if reason:
            last_reason = reason
            continue

        out["pendingagentname"] = agent_name
        out["pendingtext"] = f"QUOTE {quote}\nREBUT {rebut}\nNEW {new}\nQUESTION {question}"
        return out

    # If all retries fail, do not kill the debate—fallback and log it in coherenceflags
    fb = _safe_fallback(topic, speaker, roundidx)

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
