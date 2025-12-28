from __future__ import annotations

from typing import Any, Dict

from nodes.logger import log_event
from nodes.state import DebateState


def logger_node(state: DebateState) -> DebateState:
    log_path = state["log_path"]

    snapshot_keys = [
        "topic",
        "round_idx",
        "next_speaker",
        "status",
        "error",
        "summary",
        "coherence_flags",
        "last_node",
    ]
    snapshot: Dict[str, Any] = {k: state.get(k) for k in snapshot_keys}

    turns = state.get("turns", [])
    snapshot["turns_tail"] = turns[-2:]

    if "verdict" in state:
        snapshot["verdict"] = state["verdict"]

    log_event(log_path, {"type": "STATE_TRANSITION", "snapshot": snapshot})

    # IMPORTANT: return no state updates so we don't clobber last_node.
    return {}

