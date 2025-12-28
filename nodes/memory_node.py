from __future__ import annotations

from nodes.state import DebateState, Turn


def memory_node(state: DebateState) -> DebateState:
    if state.get("status") == "ERROR":
        return {"last_node": "MemoryNode(ERROR-PASS)"}

    speaker = state["pending_speaker"]
    agent_name = state["pending_agent_name"]
    text = state["pending_text"]

    turns = list(state.get("turns", []))
    round_no = len(turns) + 1

    turn: Turn = {
        "round": round_no,
        "agent": agent_name,
        "speaker": speaker,
        "text": text,
        "meta": {},
    }
    turns.append(turn)

    # Rolling summary: keep it short and structured
    last_two = turns[-2:]
    summary = state.get("summary", "")
    new_summary = summary
    if not new_summary:
        new_summary = f"Topic: {state['topic']}. "
    new_summary = (new_summary + f"[R{round_no}:{agent_name}] {text[:120]}... ").strip()

    # Update repetition index
    argument_norms = list(state.get("argument_norms", []))
    argument_norms.append(text)

    # Compute next speaker and memory slices (no full-state broadcast)
    next_speaker = "B" if speaker == "A" else "A"

    def slice_for(s: str):
        own = None
        opp = None
        for t in reversed(turns):
            if t["speaker"] == s and own is None:
                own = t
            if t["speaker"] != s and opp is None:
                opp = t
            if own and opp:
                break
        return {
        "summary": new_summary[-700:],
        "recent_turns": [{"round": t["round"], "agent": t["agent"], "text": t["text"]} for t in turns[-3:]],
        "last_own_turn": {"round": own["round"], "text": own["text"]} if own else None,
        "last_opponent_turn": {"round": opp["round"], "text": opp["text"]} if opp else None,
        "you_are": "AgentA" if s == "A" else "AgentB",
        }


    return {
        "turns": turns,
        "round_idx": len(turns),
        "summary": new_summary,
        "argument_norms": argument_norms,
        "next_speaker": next_speaker,
        "memory_for_a": slice_for("A"),
        "memory_for_b": slice_for("B"),
        "last_node": "MemoryNode",
    }
