from __future__ import annotations

from nodes.state import DebateState


def _sanitize_topic(topic: str) -> str:
    topic = topic.strip()
    topic = "".join(ch for ch in topic if ch.isprintable())
    return topic


def _validate_topic(topic: str) -> None:
    if len(topic) < 10:
        raise ValueError("Topic too short (min 10 characters).")
    if len(topic) > 300:
        raise ValueError("Topic too long (max 300 characters).")


def user_input_node(state: DebateState) -> DebateState:
    raw = state.get("raw_topic", "")
    topic = _sanitize_topic(raw)
    _validate_topic(topic)

    return {
        "topic": topic,
        "status": "OK",
        "error": "",
        "round_idx": 0,
        "next_speaker": "A",
        "turns": [],
        "summary": "",
        "memory_for_a": {"summary": "", "recent_turns": []},
        "memory_for_b": {"summary": "", "recent_turns": []},
        "argument_norms": [],
        "coherence_flags": [],
        "last_node": "UserInputNode",
    }
