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
    out: Dict[str, Any] = dict(state)

    raw = (out.get("rawtopic") or out.get("topic") or "").strip()
    topic = sanitize_topic(raw)
    validate_topic(topic)

    out["rawtopic"] = raw
    out["topic"] = topic

    # assignment requirement
    out["maxrounds"] = 8
    out.setdefault("maxretries", 2)
    out.setdefault("gotojudge", True)

    # reset debate fields
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

    out["memoryfora"] = {"summary": "", "recentturns": [], "lastownturn": None, "lastopponentturn": None, "youare": "AgentA"}
    out["memoryforb"] = {"summary": "", "recentturns": [], "lastownturn": None, "lastopponentturn": None, "youare": "AgentB"}

    # logging helpers
    out["last_node_io"] = {"node": "USER_INPUT", "input": {"rawtopic": raw}, "output": {"topic": topic}}
    out["last_node_name"] = "UserInputNode"

    out["lastnode"] = "USER_INPUT"
    return out





