from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(log_path: str, event: Dict[str, Any]) -> None:
    record = {
        "ts": _ts(),
        **event,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
