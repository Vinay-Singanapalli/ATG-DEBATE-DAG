from __future__ import annotations

from typing import Any, Dict

from nodes.state import DebateState


def sanitize_topic(topic: str) -> str:
    topic = (topic or "").strip()
    topic = "".join(ch for ch in topic if ch.isprintable())
    return topic


def validate_topic(topic: str) -> None:
    if len(topic) < 10:
        raise ValueError("Topic too short (min 10 characters).")
    if len(topic) > 300:
        raise ValueError("Topic too long (max 300 characters).")


def user_input_node(state: DebateState) -> DebateState:
    """
    Must preserve runtime/config keys already in state, especially:
    - logpath
    - seed
    - maxrounds/maxretries
    """
    out: Dict[str, Any] = dict(state)  # <-- preserve everything

    raw = (out.get("rawtopic") or out.get("topic") or "").strip()
    topic = sanitize_topic(raw)
    validate_topic(topic)

    out["rawtopic"] = raw
    out["topic"] = topic

    # Hard requirement
    out["maxrounds"] = 8
    out.setdefault("maxretries", 2)

    # Reset debate fields only (do NOT wipe config keys)
    out["status"] = "OK"
    out["error"] = ""
    out["verdict"] = None

    out["turns"] = []
    out["summary"] = ""

    out["roundidx"] = 0
    out["nextspeaker"] = "A"

    out["pendingspeaker"] = "A"
    out["pendingagentname"] = out.get("agentaname", "Scientist")
    out["pendingtext"] = ""

    out["retrycount"] = 0
    out["retryreason"] = ""
    out["lastrejectedtext"] = ""
    out["rejectionhistory"] = []

    out["formatviolations"] = []
    out["coherenceflags"] = []

    out["usedquotes"] = []

    # Agent memory slices (no full-state broadcast)
    out["memoryfora"] = {
        "summary": "",
        "recentturns": [],
        "lastownturn": None,
        "lastopponentturn": None,
        "youare": "AgentA",
    }
    out["memoryforb"] = {
        "summary": "",
        "recentturns": [],
        "lastownturn": None,
        "lastopponentturn": None,
        "youare": "AgentB",
    }

    out["lastnode"] = "USER_INPUT"
    return out




