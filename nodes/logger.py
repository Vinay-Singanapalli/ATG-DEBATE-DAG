from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


def ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(log_path: str, event: Dict[str, Any]) -> None:
    """
    Append JSONL event and flush immediately.
    """
    if not log_path:
        return

    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    record = {"ts": ts(), **event}
    line = json.dumps(record, ensure_ascii=False)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

