from __future__ import annotations

import json
from typing import Any, Dict

from langchain_ollama import ChatOllama
from nodes.state import DebateState


def judge_node(state: DebateState) -> DebateState:
    out: Dict[str, Any] = dict(state)

    turns = out.get("turns", [])
    topic = out.get("topic", "")

    # Transcript uses the clean paragraph text stored in turns.
    transcript = "\n".join([f"R{t.get('round')} {t.get('agent')}: {t.get('text')}" for t in turns])

    judge_model = out.get("judgemodel") or out.get("judge_model") or "llama3.2:1b"

    llm = ChatOllama(
        model=judge_model,
        temperature=0.0,
        format="json",
        num_predict=420,
    )

    system = (
        "You are an impartial debate judge.\n"
        "Return ONLY valid JSON with keys: summary, winner, reason.\n"
        "winner MUST be exactly 'Scientist' or 'Philosopher'.\n"
        "summary and reason MUST be concise strings.\n"
    )
    user = f"Topic: {topic}\nTranscript:\n{transcript}"

    msg = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    raw = getattr(msg, "content", str(msg)).strip()

    # attach coherence flags into verdict for auditability
    coherenceflags = out.get("coherenceflags", [])

    verdict: Dict[str, Any]
    try:
        parsed = json.loads(raw) if raw.startswith("{") else {}
        winner = str(parsed.get("winner", "")).strip()
        if winner not in ("Scientist", "Philosopher"):
            # fallback mapping if model used old values
            w2 = str(parsed.get("winner", "")).strip()
            if w2 == "AgentA":
                winner = "Scientist"
            elif w2 == "AgentB":
                winner = "Philosopher"
            else:
                winner = "Scientist"

        reason = str(parsed.get("reason", "")).strip()
        if not reason:
            # backward compat if model returns "justification"
            reason = str(parsed.get("justification", "")).strip()

        verdict = {
            "summary": str(parsed.get("summary", "")).strip(),
            "winner": winner,
            "reason": reason,
            "coherenceflags": coherenceflags,
        }
    except Exception:
        verdict = {
            "summary": raw[:2000],
            "winner": "Scientist",
            "reason": "Judge returned invalid JSON; raw output stored in summary.",
            "coherenceflags": coherenceflags,
        }

    out["verdict"] = verdict
    out["status"] = "OK"
    out["error"] = ""
    out["lastnode"] = "JUDGE"
    return out
