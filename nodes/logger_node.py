from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from nodes.state import DebateState


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_log_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_project_root(), "examples", f"debate_log_{ts}.jsonl")


def _safe_tail(xs: Any, n: int) -> Any:
    if isinstance(xs, list):
        return xs[-n:]
    return xs


def logger_node(state: DebateState) -> DebateState:
    out: Dict[str, Any] = dict(state)

    log_path = out.get("logpath") or out.get("log_path") or out.get("logPath")
    if not log_path:
        log_path = _default_log_path()
        out["logpath"] = log_path

    abs_path = os.path.abspath(log_path)
    out["logpath"] = abs_path
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    turns: List[Dict[str, Any]] = list(out.get("turns", []))
    coherenceflags: List[Dict[str, Any]] = list(out.get("coherenceflags", []))
    rejectionhistory: List[Dict[str, Any]] = list(out.get("rejectionhistory", []))

    # These will be added by other nodes in the next rewrite pass.
    # LoggerNode will log them if present, otherwise it logs what it can.
    node_name = out.get("lastnode", "")
    node_io = out.get("last_node_io")  # optional dict: {"node":..., "input":..., "output":...}
    node_io_name = out.get("last_node_name")

    record: Dict[str, Any] = {
        "ts": _utc_ts(),
        "type": "STATE_TRANSITION",
        "node": node_name,
        "debug": {
            "cwd": os.getcwd(),
            "has_last_node_io": node_io is not None,
            "has_last_node_name": node_io_name is not None,
        },
        "snapshot": {
            "topic": out.get("topic", ""),
            "roundidx": out.get("roundidx", 0),
            "maxrounds": out.get("maxrounds", 8),
            "nextspeaker": out.get("nextspeaker", ""),
            "pendingspeaker": out.get("pendingspeaker", ""),
            "status": out.get("status", ""),
            "error": out.get("error", ""),
            "lastnode": out.get("lastnode", ""),
            "turns_len": len(turns),
            "coherenceflags_len": len(coherenceflags),
            "rejectionhistory_len": len(rejectionhistory),
        },
        # Keep logs readable: store only tails
        "turns_tail": _safe_tail(turns, 2),
        "coherenceflags_tail": _safe_tail(coherenceflags, 6),
        "rejectionhistory_tail": _safe_tail(rejectionhistory, 3),
    }

    if isinstance(node_io, dict):
        record["node_io"] = node_io
    if isinstance(node_io_name, str) and node_io_name.strip():
        record["node_io_name"] = node_io_name

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

    if not out.get("lastnode"):
        out["lastnode"] = "LOGGER"

    return out


