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


def default_dag_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(project_root(), "examples", f"debate_dag_{ts}.png")


def first_present(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _format_one_line(text: str) -> str:
    return " ".join((text or "").split())


def _agent_display_name(state: Dict[str, Any], speaker: str) -> str:
    if speaker == "A":
        return state.get("agentaname", "Scientist")
    return state.get("agentbname", "Philosopher")


def _strip_struct_labels(text: str) -> str:
    """
    Converts internal structured turn format:
      QUOTE ...
      REBUT ...
      NEW ...
      QUESTION ...
    into a single paragraph suitable for CLI display.
    """
    lines = [(ln or "").strip() for ln in (text or "").splitlines() if (ln or "").strip()]
    parts = []
    for ln in lines:
        up = ln.upper()
        if up.startswith("QUOTE "):
            # In the sample CLI, quotes are not shown
            continue
        if up.startswith("REBUT "):
            parts.append(ln.split(" ", 1)[1].strip())
            continue
        if up.startswith("NEW "):
            parts.append(ln.split(" ", 1)[1].strip())
            continue
        if up.startswith("QUESTION "):
            # In the sample CLI, questions are not shown
            continue
        parts.append(ln)
    return _format_one_line(" ".join(parts))


def _turn_to_cli_text(turn: Dict[str, Any]) -> str:
    # Prefer explicit display fields if your nodes provide them (we will add later)
    for k in ("display", "display_text", "spoken", "final"):
        v = turn.get(k)
        if isinstance(v, str) and v.strip():
            return _format_one_line(v)

    # Fallback to transforming your internal text
    return _strip_struct_labels(turn.get("text", ""))


def _try_write_dag(app: Any, dag_path: str) -> None:
    """
    Best-effort DAG export; will not crash if unsupported.
    Many LangGraph apps support app.get_graph() then drawing via Mermaid. [web:102][web:238]
    """
    try:
        g = app.get_graph()
    except Exception:
        return

    # Try common draw methods
    try:
        png_bytes = g.draw_mermaid_png()
        os.makedirs(os.path.dirname(os.path.abspath(dag_path)), exist_ok=True)
        with open(dag_path, "wb") as f:
            f.write(png_bytes)
        return
    except Exception:
        pass

    # Some setups expose draw_png instead (varies by version)
    try:
        png_bytes = g.draw_png()
        os.makedirs(os.path.dirname(os.path.abspath(dag_path)), exist_ok=True)
        with open(dag_path, "wb") as f:
            f.write(png_bytes)
        return
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default=None, help="Debate topic; if omitted, you'll be prompted.")
    p.add_argument("--seed", type=int, default=None, help="Optional seed (best-effort determinism).")
    p.add_argument("--log-path", default=None, help="Path to JSONL log file.")
    p.add_argument("--dag-path", default=None, help="Path to DAG PNG output (optional).")
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

    dag_path = args.dag_path or default_dag_path()
    if not os.path.isabs(dag_path):
        dag_path = os.path.join(project_root(), dag_path)

    app = build_graph()

    # Best-effort DAG export (won't fail run if unsupported)
    _try_write_dag(app, dag_path)

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
        "last_node_io": {},
        "last_node_name": "",

    }

    # Sample-style intro
    a_name = init_state.get("agentaname", "Scientist")
    b_name = init_state.get("agentbname", "Philosopher")
    print(f"Starting debate between {a_name} and {b_name}...")
    print(f"Starting debate on: {topic}")
    print(f"Log file: {log_path}\n")
    print(f"DAG: {dag_path}\n")

    last_seen_turns_len = 0
    final_state: Dict[str, Any] = init_state

    # Stream state updates after each node
    for chunk in app.stream(
        init_state,
        stream_mode="updates",
        config={"recursion_limit": int(args.recursion_limit)},
    ):
        if not isinstance(chunk, dict) or not chunk:
            continue

        node_name, update = next(iter(chunk.items()))
        if not isinstance(update, dict):
            continue

        final_state = {**final_state, **update}

        # Print when a new turn is appended (typically by MemoryNode)
        if node_name == "MemoryNode" and "turns" in update:
            turns = update.get("turns") or []
            if isinstance(turns, list) and len(turns) > last_seen_turns_len:
                for t in turns[last_seen_turns_len:]:
                    r = t.get("round")
                    speaker = t.get("speaker")
                    agent_name = t.get("agent") or _agent_display_name(final_state, speaker)
                    text = _turn_to_cli_text(t)
                    print(f"[Round {r}] {agent_name}: {text}")
                last_seen_turns_len = len(turns)

        if update.get("status") == "ERROR":
            print("\n[ERROR]", update.get("error", "Unknown error"))
            break

    print("\n[Judge]")
    verdict = final_state.get("verdict") or {}
    if isinstance(verdict, dict):
        if verdict.get("summary"):
            print("Summary of debate:")
            print(verdict["summary"])
        if verdict.get("winner"):
            print("Winner:", verdict["winner"])
        if verdict.get("reason"):
            print("Reason:", verdict["reason"])
        # Backward compatible with your earlier verdict format
        if verdict.get("justification") and not verdict.get("reason"):
            print("Reason:", verdict["justification"])

    print("\nDone.")
    print("Log:", final_state.get("logpath", log_path))
    print("Final round:", final_state.get("roundidx"))
    print("Last node:", final_state.get("lastnode"))


if __name__ == "__main__":
    main()
