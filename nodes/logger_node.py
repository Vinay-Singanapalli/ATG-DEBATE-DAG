from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from nodes.state import DebateState


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> str:
    # nodes/.. (repo root where run_debate.py typically lives)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_log_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_project_root(), "examples", f"debate_log_{ts}.jsonl")


def logger_node(state: DebateState) -> DebateState:
    out: Dict[str, Any] = dict(state)

    # Accept multiple spellings to avoid “missing logpath” from older nodes
    log_path = out.get("logpath") or out.get("log_path") or out.get("logPath")

    # Self-heal: if missing, generate one so logging always works
    if not log_path:
        log_path = _default_log_path()
        out["logpath"] = log_path

    abs_path = os.path.abspath(log_path)
    out["logpath"] = abs_path  # normalize

    # Ensure directory exists
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    # Build event
    turns = list(out.get("turns", []))
    record: Dict[str, Any] = {
        "ts": _utc_ts(),
        "type": "STATE_TRANSITION",
        "debug": {
            "has_logpath": "logpath" in out,
            "has_log_path": "log_path" in out,
            "has_logPath": "logPath" in out,
            "cwd": os.getcwd(),
        },
        "snapshot": {
            "topic": out.get("topic", ""),
            "roundidx": out.get("roundidx", 0),
            "nextspeaker": out.get("nextspeaker", ""),
            "pendingspeaker": out.get("pendingspeaker", ""),
            "status": out.get("status", ""),
            "error": out.get("error", ""),
            "lastnode": out.get("lastnode", ""),
            "turns_len": len(turns),
        },
    }

    # Append JSONL (create file if missing)
    try:
        with open(abs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        out["status"] = "ERROR"
        out["error"] = f"LoggerNode exception: {type(e).__name__}: {e}"
        out["lastnode"] = "LOGGER"
        return out

    # Keep lastnode unchanged; but if empty, mark LOGGER for debugging
    if not out.get("lastnode"):
        out["lastnode"] = "LOGGER"

    return out






