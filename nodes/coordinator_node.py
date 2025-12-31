from __future__ import annotations

from typing import Any, Dict

from nodes.state import DebateState


def coordinator_node(state: DebateState) -> DebateState:
    """
    Round controller / coordinator:
    - Enforces sequencing
    - Sets pending speaker + pending agent name for the next agent node
    - Sends to judge after exactly 8 turns (handled via lastnode for router)
    """
    out: Dict[str, Any] = dict(state)

    if out.get("status") == "ERROR":
        out["lastnode"] = "COORDINATOR"
        return out

    # Hard requirement
    max_rounds = int(out.get("maxrounds", 8))
    if max_rounds != 8:
        out["status"] = "ERROR"
        out["error"] = "Assignment requires exactly 8 rounds (maxrounds must be 8)."
        out["lastnode"] = "COORDINATOR"
        return out

    turns = list(out.get("turns", []))
    out["roundidx"] = int(out.get("roundidx", len(turns)))

    # Stop after 8 turns
    if out["roundidx"] >= 8:
        out["lastnode"] = "COORDINATOR_TO_JUDGE"
        return out

    nextspeaker = out.get("nextspeaker", "A")
    if nextspeaker not in ("A", "B"):
        out["status"] = "ERROR"
        out["error"] = f"Invalid nextspeaker: {nextspeaker}"
        out["lastnode"] = "COORDINATOR"
        return out

    # These keys MUST match what Agent nodes read
    out["pendingspeaker"] = nextspeaker
    out["pendingagentname"] = out.get("agentaname", "Scientist") if nextspeaker == "A" else out.get("agentbname", "Philosopher")
    out["pendingtext"] = ""

    out["lastnode"] = "COORDINATOR"
    return out


