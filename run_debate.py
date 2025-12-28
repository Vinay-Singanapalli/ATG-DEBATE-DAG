from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from nodes.graph_builder import build_graph
from nodes.state import DebateState


def build_log_path(log_path: str | None, default_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_path:
        return log_path
    Path(default_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(default_dir) / f"debate_log_{ts}.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    max_rounds = int(cfg["debate"]["max_rounds"])

    raw_topic = input("Enter topic for debate: ")

    log_path = build_log_path(args.log_path, cfg["logging"]["default_log_dir"])

    state: DebateState = {
        "raw_topic": raw_topic,
        "max_rounds": max_rounds,
        "seed": args.seed,
        "log_path": log_path,
        "status": "OK",
        "error": "",
    }

    state.update({
        "llm_model": "llama3.2:1b",
        "judge_model": "llama3.2:1b",
        "embed_model": "mxbai-embed-large",
        "llm_temperature": 0.2,
        "llm_max_tokens": 180,
        "repetition_max_cosine": 0.97,
        "topic_min_cosine": 0.40,
        "argument_embeddings": [],
        "topic_embedding": [],
        "format_violations": [],
        })





    graph = build_graph().compile(checkpointer=MemorySaver())  # easy persistence for dev [web:40]
    config = RunnableConfig(
    configurable={"thread_id": "debate_cli"},
    recursion_limit=200,  # enough for 8 rounds with multiple nodes per round
    )


    print(f"Starting debate between Scientist and Philosopher... (log: {log_path})")

    final = graph.invoke(state, config=config)

    # Print the round-by-round output to stdout (required)
    turns = final.get("turns", [])
    for t in turns:
        print(f"[Round {t['round']}] {t['agent']}: {t['text']}")

    verdict = final.get("verdict", {})
    print("[Judge] Summary of debate:")
    print(verdict.get("summary", "(missing summary)"))
    print(f"[Judge] Winner: {verdict.get('winner', '(missing winner)')}")
    print(f"Reason: {verdict.get('justification', '(missing justification)')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

