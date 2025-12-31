from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Optional

from nodes.graph_builder import build_graph


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_log_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(project_root(), "examples", f"debate_log_{ts}.jsonl")


def first_present(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _format_one_line(text: str) -> str:
    return " ".join((text or "").split())


def _agent_display_name(state: Dict[str, Any], speaker: str) -> str:
    if speaker == "A":
        return state.get("agentaname", "AgentA")
    return state.get("agentbname", "AgentB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default=None, help="Debate topic; if omitted, you'll be prompted.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log-path", default=None, help="Path to JSONL log file.")
    p.add_argument("--max-rounds", type=int, default=8, help="Must be 8 for this assignment.")
    p.add_argument("--recursion-limit", type=int, default=200, help="LangGraph recursion limit.")
    args = p.parse_args()

    topic = args.topic
    if not topic:
        try:
            topic = input("Enter topic for debate: ").strip()
        except EOFError:
            topic = ""

    if not topic:
        raise SystemExit("Error: topic is required.")

    if args.max_rounds != 8:
        raise SystemExit("Error: This assignment requires exactly 8 rounds. Use --max-rounds 8.")

    log_path = args.log_path or default_log_path()
    if not os.path.isabs(log_path):
        log_path = os.path.join(project_root(), log_path)

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    app = build_graph()

    init_state: Dict[str, Any] = {
        "rawtopic": topic,
        "topic": topic,
        "maxrounds": 8,
        "maxretries": 2,
        "logpath": log_path,
        "seed": args.seed,
        "gotojudge": True,

        "status": "OK",
        "error": "",
        "turns": [],
        "summary": "",
        "verdict": None,

        "roundidx": 0,
        "nextspeaker": "A",

        "pendingspeaker": "A",
        "pendingagentname": "",
        "pendingtext": "",

        "coherenceflags": [],
        "formatviolations": [],
        "rejectionhistory": [],
        "retrycount": 0,
        "retryreason": "",
        "lastrejectedtext": "",
        "usedquotes": [],
        "lastnode": "",
    }

    print(f"Starting debate on: {topic}")
    print(f"Log file: {log_path}\n")

    last_seen_turns_len = 0
    final_state: Dict[str, Any] = init_state

    # Stream state updates after each node
    for chunk in app.stream(
        init_state,
        stream_mode="updates",
        config={"recursion_limit": int(args.recursion_limit)},
    ):
        # chunk looks like {"NodeName": {<state updates>}}
        if not isinstance(chunk, dict) or not chunk:
            continue

        node_name, update = next(iter(chunk.items()))
        if not isinstance(update, dict):
            continue

        # Keep an accumulated final_state (best effort merge)
        final_state = {**final_state, **update}

        # If MemoryNode updated turns, print the new turn(s)
        # (MemoryNode is where your turn gets appended and roundidx advances)
        if node_name == "MemoryNode" and "turns" in update:
            turns = update.get("turns") or []
            if isinstance(turns, list) and len(turns) > last_seen_turns_len:
                for t in turns[last_seen_turns_len:]:
                    r = t.get("round")
                    speaker = t.get("speaker")
                    agent_name = t.get("agent") or _agent_display_name(final_state, speaker)
                    text = _format_one_line(t.get("text", ""))
                    print(f"[Round {r}] {agent_name}: {text}")
                last_seen_turns_len = len(turns)

        # If any node sets an error, print and stop streaming early
        if update.get("status") == "ERROR":
            print("\n[ERROR]", update.get("error", "Unknown error"))
            break

    print("\n[Judge]")
    verdict = final_state.get("verdict") or {}
    if isinstance(verdict, dict):
        if verdict.get("summary"):
            print("Summary:", verdict["summary"])
        if verdict.get("winner"):
            print("Winner:", verdict["winner"])
        if verdict.get("justification"):
            print("Justification:", verdict["justification"])

    print("\nDone.")
    print("Log:", final_state.get("logpath", log_path))
    print("Final round:", final_state.get("roundidx"))
    print("Last node:", final_state.get("lastnode"))


if __name__ == "__main__":
    main()

