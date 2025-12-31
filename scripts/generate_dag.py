from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so "import nodes" works even when running:
#   python scripts/generate_dag.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nodes.graph_builder import build_graph


def main() -> int:
    out = ROOT / "dag.png"  # always write to repo root
    graph = build_graph().compile()
    png_bytes = graph.get_graph().draw_mermaid_png()
    out.write_bytes(png_bytes)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
