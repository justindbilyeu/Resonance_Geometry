#!/usr/bin/env python3
"""
Phase-surface loader for Resonance Geometry theory artifacts.

- Loads any of:
    docs/data/theory/phase_surface_stable.json
    docs/data/theory/phase_surface_balanced.json
    docs/data/theory/phase_surface_divergent.json
    docs/data/theory/phase_surface_all.json

- Returns a normalized dict with:
    {
      "regimes": {
         "stable":   {"count": int, "alpha_over_beta": float|None},
         "balanced": {"count": int, "alpha_over_beta": float|None},
         "divergent":{"count": int, "alpha_over_beta": float|None}
      },
      "files_present": [...],
      "total_rows": int
    }

- Tolerates missing files and heterogeneous schemas (older dumps).
- No heavy deps; pure stdlib.

Usage (CLI):
    python -m tools.load_phase_surface --print
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Dict, Any, List

THEORY_DIR = Path("docs/data/theory")
CANDIDATES = [
    "phase_surface_stable.json",
    "phase_surface_balanced.json",
    "phase_surface_divergent.json",
    "phase_surface_all.json",
]

def _load_json(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            data = json.load(f)
        # Accept either list-of-rows or dict with 'rows'
        if isinstance(data, dict) and "rows" in data:
            return data["rows"]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def _infer_regime(row: Dict[str, Any]) -> str:
    # Prefer explicit label if present
    for k in ("regime", "stability_flag"):
        if isinstance(row.get(k), str):
            lab = row[k].strip().lower()
            if lab in ("stable","grounded"):   return "stable"
            if lab in ("balanced","creative"): return "balanced"
            if lab in ("divergent","hallucinatory"): return "divergent"
    # Heuristic fallback via alpha/beta if provided
    aob = row.get("alpha_over_beta")
    if isinstance(aob, (int,float)):
        if aob < 1.0:  return "stable"
        if aob > 1.0:  return "divergent"
        return "balanced"
    # Last resort
    return "balanced"

def load_phase_surfaces(base_dir: Path = THEORY_DIR) -> Dict[str, Any]:
    present: List[str] = []
    rows: List[Dict[str, Any]] = []

    for name in CANDIDATES:
        p = base_dir / name
        if p.exists():
            present.append(name)
            rows.extend(_load_json(p))

    # Aggregate counts by regime
    counts = {"stable": 0, "balanced": 0, "divergent": 0}
    aob_seen = {"stable": [], "balanced": [], "divergent": []}

    for r in rows:
        regime = _infer_regime(r)
        counts[regime] += 1
        aob = r.get("alpha_over_beta")
        if isinstance(aob, (int, float)):
            aob_seen[regime].append(float(aob))

    def _mean(xs: List[float]) -> float|None:
        return sum(xs)/len(xs) if xs else None

    out = {
        "regimes": {
            "stable":   {"count": counts["stable"],   "alpha_over_beta": _mean(aob_seen["stable"])},
            "balanced": {"count": counts["balanced"], "alpha_over_beta": _mean(aob_seen["balanced"])},
            "divergent":{"count": counts["divergent"],"alpha_over_beta": _mean(aob_seen["divergent"])},
        },
        "files_present": present,
        "total_rows": sum(counts.values()),
    }
    return out

def main(argv: List[str]) -> int:
    data = load_phase_surfaces()
    if "--print" in argv or not argv:
        print(json.dumps(data, indent=2))
    else:
        # Optional: write a small status sidecar if requested
        if "--write" in argv:
            out = Path("docs/data/theory/phase_surface_status.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data, indent=2))
            print(f"[theory] wrote {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
