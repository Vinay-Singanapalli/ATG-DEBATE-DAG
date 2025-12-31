from __future__ import annotations

from typing import Any, Dict

from nodes.state import DebateState


def coordinator_node(state: DebateState) -> DebateState:
    out: Dict[str, Any] = dict(state)

    if out.get("status") == "ERROR":
        out["lastnode"] = "COORDINATOR"
        out["last_node_io"] = {"node": "COORDINATOR", "input": {}, "output": {"status": "ERROR"}}
        out["last_node_name"] = "Coordinator"
        return out

    turns = list(out.get("turns", []))
    out["roundidx"] = int(out.get("roundidx", len(turns)))

    max_rounds = int(out.get("maxrounds", 8))
    if out["roundidx"] >= max_rounds:
        out["lastnode"] = "COORDINATOR_TO_JUDGE"
        out["last_node_io"] = {"node": "COORDINATOR_TO_JUDGE", "input": {"roundidx": out["roundidx"]}, "output": {}}
        out["last_node_name"] = "Coordinator"
        return out

    nextspeaker = out.get("nextspeaker", "A")
    if nextspeaker not in ("A", "B"):
        out["status"] = "ERROR"
        out["error"] = f"Invalid nextspeaker: {nextspeaker}"
        out["lastnode"] = "COORDINATOR"
        out["last_node_io"] = {"node": "COORDINATOR", "input": {"nextspeaker": nextspeaker}, "output": {"status": "ERROR"}}
        out["last_node_name"] = "Coordinator"
        return out

    out["pendingspeaker"] = nextspeaker
    out["pendingagentname"] = out.get("agentaname", "Scientist") if nextspeaker == "A" else out.get("agentbname", "Philosopher")
    out["pendingtext"] = ""

    out["lastnode"] = "COORDINATOR"
    out["last_node_io"] = {
        "node": "COORDINATOR",
        "input": {"roundidx": out["roundidx"], "nextspeaker": nextspeaker},
        "output": {"pendingspeaker": out["pendingspeaker"], "pendingagentname": out["pendingagentname"]},
    }
    out["last_node_name"] = "Coordinator"
    return out


