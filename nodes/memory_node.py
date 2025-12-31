from __future__ import annotations

from typing import Any, Dict, List

from nodes.state import DebateState


def memory_node(state: DebateState) -> DebateState:
    """
    Updates:
    - turns[] transcript
    - roundidx
    - nextspeaker (flip A<->B)
    - memoryfora/memoryforb slices (no full broadcast)
    """
    out: Dict[str, Any] = dict(state)

    if out.get("status") == "ERROR":
        out["lastnode"] = "MEMORY"
        return out

    topic = out.get("topic", "")
    turns: List[Dict[str, Any]] = list(out.get("turns", []))

    speaker = out.get("pendingspeaker")
    agent_name = out.get("pendingagentname")
    text = out.get("pendingtext") or ""

    if speaker not in ("A", "B") or not agent_name or not text.strip():
        out["status"] = "ERROR"
        out["error"] = "MemoryNode missing pending speaker/agent/text."
        out["lastnode"] = "MEMORY"
        return out

    round_no = len(turns) + 1
    turns.append(
        {
            "round": round_no,
            "agent": agent_name,
            "speaker": speaker,
            "text": text,
            "meta": {},
        }
    )

    out["turns"] = turns
    out["roundidx"] = len(turns)

    # Very small running summary (kept short)
    prev_summary = (out.get("summary") or "").strip()
    snippet = text.replace("\n", " ")[:160].strip()
    if not prev_summary:
        out["summary"] = f"Topic: {topic}. R{round_no} {agent_name}: {snippet}"
    else:
        out["summary"] = (prev_summary + f" | R{round_no} {agent_name}: {snippet}")[-900:]

    # Flip next speaker
    out["nextspeaker"] = "B" if speaker == "A" else "A"

    # Build per-agent memory slices (summary + last 3 turns + last own/opp)
    def last_turn_for(s: str) -> Dict[str, Any] | None:
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

    # Clear pending fields
    out["pendingspeaker"] = out["nextspeaker"]
    out["pendingagentname"] = ""
    out["pendingtext"] = ""

    out["lastnode"] = "MEMORY"
    return out
