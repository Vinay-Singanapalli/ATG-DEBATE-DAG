from __future__ import annotations

import json

from langchain_ollama import ChatOllama
from nodes.state import DebateState


def judge_node(state: DebateState) -> DebateState:
    turns = state.get("turns", [])
    topic = state.get("topic", "")

    transcript = "\n".join([f"R{t['round']} {t['agent']}: {t['text']}" for t in turns])

    # Force JSON output mode (Ollama supports format="json"). [web:269]
    llm = ChatOllama(
        model=state.get("judge_model", "llama3.2:1b"),
        temperature=0.0,
        format="json",
        num_predict=420,
    )

    system = (
        "You are an impartial debate judge.\n"
        "Return ONLY valid JSON with keys: summary, winner, justification.\n"
        "winner MUST be exactly 'AgentA' or 'AgentB'.\n"
        "summary and justification MUST be strings.\n"
    )
    user = f"Topic: {topic}\nTranscript:\n{transcript}"

    msg = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    raw = getattr(msg, "content", str(msg)).strip()

    # Parse JSON; if model fails, wrap raw as strings
    try:
        parsed = json.loads(raw)
        verdict = {
            "summary": str(parsed.get("summary", "")),
            "winner": str(parsed.get("winner", "")),
            "justification": str(parsed.get("justification", "")),
            "coherence_flags": state.get("coherence_flags", []),
        }
    except Exception:
        verdict = {
            "summary": str(raw[:2000]),
            "winner": "AgentA",
            "justification": "Judge returned invalid JSON; raw output stored in summary.",
            "coherence_flags": state.get("coherence_flags", []),
        }

    return {"last_node": "JudgeNode", "verdict": verdict}



