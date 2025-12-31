from __future__ import annotations

import json
from langchain_ollama import ChatOllama
from nodes.state import DebateState


def judge_node(state: DebateState) -> DebateState:
    out = dict(state)  # IMPORTANT: preserve whole state (turns, embeddings, etc.)

    turns = out.get("turns", [])
    topic = out.get("topic", "")

    transcript = "\n".join([f"R{t.get('round')} {t.get('agent')}: {t.get('text')}" for t in turns])

    llm = ChatOllama(
        model=out.get("judge_model", "llama3.2:1b"),
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

    coherence_flags = out.get("coherence_flags", out.get("coherenceflags", []))

    try:
        parsed = json.loads(raw)
        winner = str(parsed.get("winner", "")).strip()
        if winner not in ("AgentA", "AgentB"):
            winner = "AgentA"

        verdict = {
            "summary": str(parsed.get("summary", "")).strip(),
            "winner": winner,
            "justification": str(parsed.get("justification", "")).strip(),
            "coherence_flags": coherence_flags,
        }
    except Exception:
        verdict = {
            "summary": str(raw[:2000]),
            "winner": "AgentA",
            "justification": "Judge returned invalid JSON; raw output stored in summary.",
            "coherence_flags": coherence_flags,
        }

    out["verdict"] = verdict
    out["status"] = "OK"
    out["error"] = ""
    out["last_node"] = "JudgeNode"
    out["lastnode"] = "JudgeNode"
    return out




