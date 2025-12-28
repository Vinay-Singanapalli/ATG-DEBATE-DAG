from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from nodes.state import DebateState
from nodes.user_input_node import user_input_node
from nodes.coordinator_node import coordinator_node
from nodes.agent_node import agent_node
from nodes.memory_node import memory_node
from nodes.judge_node import judge_node
from nodes.logger_node import logger_node


def build_graph():
    g = StateGraph(DebateState)

    g.add_node("UserInputNode", user_input_node)
    g.add_node("Coordinator", coordinator_node)
    g.add_node("Agent", agent_node)
    g.add_node("MemoryNode", memory_node)
    g.add_node("JudgeNode", judge_node)
    g.add_node("LoggerNode", logger_node)

    g.set_entry_point("UserInputNode")

    # Always log after each main node.
    g.add_edge("UserInputNode", "LoggerNode")
    g.add_edge("Coordinator", "LoggerNode")
    g.add_edge("Agent", "LoggerNode")
    g.add_edge("MemoryNode", "LoggerNode")
    g.add_edge("JudgeNode", "LoggerNode")

    # Logger routes based on what just happened (stored in last_node).
    def route_from_logger(state: DebateState) -> Literal["Coordinator", "Agent", "MemoryNode", "JudgeNode", "__end__"]:
        if state.get("status") == "ERROR":
            return "__end__"

        last = state.get("last_node", "")

        if last == "UserInputNode":
            return "Coordinator"

        if last.startswith("Coordinator"):
            # Coordinator decided whether to go to Agent or Judge
            if last == "Coordinator->Judge":
                return "JudgeNode"
            return "Agent"

        if last.startswith("Agent"):
            return "MemoryNode"

        if last == "MemoryNode":
            return "Coordinator"

        if last == "JudgeNode":
            return "__end__"

        # Safe fallback
        return "__end__"

    g.add_conditional_edges("LoggerNode", route_from_logger, {
        "Coordinator": "Coordinator",
        "Agent": "Agent",
        "MemoryNode": "MemoryNode",
        "JudgeNode": "JudgeNode",
        "__end__": END,
    })

    return g

