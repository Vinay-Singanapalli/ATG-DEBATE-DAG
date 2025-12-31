from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from nodes.state import DebateState
from nodes.user_input_node import user_input_node
from nodes.coordinator_node import coordinator_node
from nodes.agent_node import agent_a_node, agent_b_node
from nodes.memory_node import memory_node
from nodes.judge_node import judge_node
from nodes.logger_node import logger_node


def build_graph():
    g: StateGraph = StateGraph(DebateState)

    g.add_node("UserInputNode", user_input_node)
    g.add_node("Coordinator", coordinator_node)
    g.add_node("AgentA", agent_a_node)
    g.add_node("AgentB", agent_b_node)
    g.add_node("MemoryNode", memory_node)
    g.add_node("JudgeNode", judge_node)
    g.add_node("LoggerNode", logger_node)

    g.set_entry_point("UserInputNode")

    # Always log after each node
    g.add_edge("UserInputNode", "LoggerNode")
    g.add_edge("Coordinator", "LoggerNode")
    g.add_edge("AgentA", "LoggerNode")
    g.add_edge("AgentB", "LoggerNode")
    g.add_edge("MemoryNode", "LoggerNode")
    g.add_edge("JudgeNode", "LoggerNode")

    def route_from_logger(
        state: DebateState,
    ) -> Literal["Coordinator", "AgentA", "AgentB", "MemoryNode", "JudgeNode", "end"]:
        if state.get("status") == "ERROR":
            return "end"

        lastnode = (state.get("lastnode") or "").strip().upper()

        if lastnode == "USER_INPUT":
            return "Coordinator"

        if lastnode == "COORDINATOR":
            # Route from nextspeaker (source of truth). Coordinator sets pendingspeaker too.
            ns = state.get("nextspeaker", "A")
            return "AgentA" if ns == "A" else "AgentB"

        if lastnode in ("AGENT_A", "AGENT_B"):
            return "MemoryNode"

        if lastnode == "MEMORY":
            if int(state.get("roundidx", 0)) >= 8:
                return "JudgeNode" if state.get("gotojudge", True) else "end"
            return "Coordinator"

        if lastnode == "COORDINATOR_TO_JUDGE":
            return "JudgeNode" if state.get("gotojudge", True) else "end"

        if lastnode == "JUDGE":
            return "end"

        return "end"

    g.add_conditional_edges(
        "LoggerNode",
        route_from_logger,
        {
            "Coordinator": "Coordinator",
            "AgentA": "AgentA",
            "AgentB": "AgentB",
            "MemoryNode": "MemoryNode",
            "JudgeNode": "JudgeNode",
            "end": END,
        },
    )

    return g.compile()



