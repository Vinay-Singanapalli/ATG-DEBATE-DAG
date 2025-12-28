from __future__ import annotations

from nodes.state import DebateState


def coordinator_node(state: DebateState) -> DebateState:
    if state.get("status") == "ERROR":
        return {"last_node": "Coordinator(ERROR-PASS)"}

    max_rounds = int(state["max_rounds"])
    round_idx = int(state.get("round_idx", 0))

    # If we've completed exactly max_rounds turns, coordinator should route to Judge.
    if round_idx >= max_rounds:
        return {"last_node": "Coordinator->Judge"}

    # Otherwise, route to the next agent deterministically.
    next_speaker = state.get("next_speaker", "A")
    if next_speaker not in ("A", "B"):
        return {"status": "ERROR", "error": f"Invalid next_speaker={next_speaker}", "last_node": "Coordinator(ERROR)"}

    agent_name = "Scientist" if next_speaker == "A" else "Philosopher"
    return {
        "pending_speaker": next_speaker,
        "pending_agent_name": agent_name,
        "last_node": "Coordinator->Agent",
    }
