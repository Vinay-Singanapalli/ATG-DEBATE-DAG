from __future__ import annotations

from pathlib import Path

from nodes.graph_builder import build_graph


def main() -> int:
    out = Path("dag.png")
    graph = build_graph().compile()
    png_bytes = graph.get_graph().draw_mermaid_png()  # standard LangGraph visualization pattern [web:5]
    out.write_bytes(png_bytes)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
