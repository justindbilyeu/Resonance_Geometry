#!/usr/bin/env python3
"""Aggregate experiment summaries into a unified dashboard."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, Iterable, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
STATUS = DATA / "status" / "summary.json"
OUT_JSON = DATA / "overview.json"
OUT_HTML = DOCS / "assets" / "fragment.html"


def _load_json(path: Optional[pathlib.Path]) -> Optional[Any]:
    if not path:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _first_existing(paths: Iterable[pathlib.Path]) -> Optional[pathlib.Path]:
    for path in paths:
        if path and path.exists():
            return path
    return None


def _count_lines(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _render_card(title: str, body: str) -> str:
    return f'<div class="card"><h3>{title}</h3><div class="card-body">{body}</div></div>'


def main() -> None:
    overview: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": {
            "current": "Phase 2/3 (Jacobian + Fluency)",
            "next": "Phase 4 (Unified Dashboard)",
        },
        "experiments": {},
    }

    status = _load_json(STATUS) or {}

    pilot_null = _load_json(DATA / "pilot_null" / "summary.json")
    pilot_proxy = _load_json(DATA / "pilot_proxy" / "summary.json")
    overview["experiments"]["phase1_pilots"] = {
        "null": pilot_null,
        "proxy": pilot_proxy,
    }

    ring_src = _first_existing(
        [
            DATA / "ringing_sweep" / "summary.json",
            ROOT / "results" / "ringing_sweep" / "summary.json",
        ]
    )
    overview["experiments"]["ringing_sweep"] = _load_json(ring_src)

    jac_boundary_path = ROOT / "results" / "jacobian_sweep" / "boundary.json"
    jac_grid_path = ROOT / "results" / "jacobian_sweep" / "grid.jsonl"
    jac_meta_path = ROOT / "results" / "jacobian_sweep" / "meta.json"

    jac_boundary = _load_json(jac_boundary_path) or []
    overview["experiments"]["jacobian"] = {
        "boundary_points": jac_boundary,
        "meta": _load_json(jac_meta_path),
        "stats": {
            "n_boundary": len(jac_boundary),
            "grid_entries": _count_lines(jac_grid_path),
        },
    }

    flu_path = _first_existing(
        [
            DATA / "fluency_sweep" / "summary.json",
            ROOT / "results" / "fluency_sweep" / "summary.json",
        ]
    )
    overview["experiments"]["fluency_sweep"] = _load_json(flu_path)

    overview["status"] = status

    _ensure_parent(OUT_JSON)
    OUT_JSON.write_text(json.dumps(overview, indent=2) + "\n", encoding="utf-8")

    cards = []
    ringing_summary = overview["experiments"].get("ringing_sweep") or {}
    ringing_fraction = None
    if isinstance(ringing_summary, dict):
        ringing_fraction = ringing_summary.get("ringing_frac")

    if isinstance(ringing_fraction, (int, float)):
        ring_body = (
            f"<p>Ringing fraction: <b>{ringing_fraction:.2f}</b></p>"
            "<p><code>docs/data/ringing_sweep/summary.json</code> or "
            "<code>results/ringing_sweep/summary.json</code></p>"
        )
    else:
        ring_body = "<p>No ringing summary found.</p>"
    cards.append(_render_card("Ringing Boundary", ring_body))

    jac_stats = overview["experiments"]["jacobian"]["stats"]
    jac_body = (
        "<p>Boundary points: <b>{n_boundary}</b><br/>"
        "Grid entries: <b>{grid_entries}</b></p>"
        "<p><code>results/jacobian_sweep/</code></p>"
    ).format(**jac_stats)
    cards.append(_render_card("Jacobian Sweep", jac_body))

    flu_summary = overview["experiments"].get("fluency_sweep")
    if isinstance(flu_summary, dict) and flu_summary:
        flu_body = "<p>Available: <b>yes</b></p>"
    else:
        flu_body = "<p>Not yet available.</p>"
    cards.append(_render_card("Fluency Velocity", flu_body))

    _ensure_parent(OUT_HTML)
    OUT_HTML.write_text("\n".join(cards) + "\n", encoding="utf-8")

    print(f"[dashboard] Wrote {OUT_JSON} and {OUT_HTML}")


if __name__ == "__main__":
    main()
