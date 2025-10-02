# experiments/gp_ringing_demo.py
"""
Lightweight GP ringing demo with a CI guard.

- Defines sane defaults at module import (so no NameError on STEPS, etc).
- When RG_CI=1 (set in GitHub Actions), clamps work to a tiny budget.
- CLI lets you override steps/runs/seeds/grid/out.

This script is intentionally lightweight so it won’t blow up smoke runs.
You can replace `run_demo()` internals later with the heavy version.
"""

from __future__ import annotations
import os
import json
import time
import argparse
from pathlib import Path
from typing import Tuple

# ---------- defaults available at import time ----------
STEPS: int = 2000
RUNS: int = 30000
SEEDS: int = 5
GRID: Tuple[int, int, int, int] = (8, 8, 8, 8)
OUT_DIR: str = "results/gp_ringing_demo"

# Trim automatically in CI
if os.getenv("RG_CI", "0") == "1":
    STEPS = min(STEPS, 200)
    RUNS = min(RUNS, 2000)
    SEEDS = min(SEEDS, 1)


def run_demo(steps: int, runs: int, seeds: int, grid: Tuple[int, int, int, int], out_dir: str) -> dict:
    """
    Replace this stub with the real ringing demo. For CI, just write a tiny artifact.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Minimal, deterministic-ish payload
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": int(steps),
        "runs": int(runs),
        "seeds": int(seeds),
        "grid": list(map(int, grid)),
        "note": "CI-safe ringing demo stub (replace with real compute as needed).",
    }
    with open(Path(out_dir) / "ringing_demo_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GP ringing demo (CI-safe).")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--runs", type=int, default=RUNS)
    p.add_argument("--seeds", type=int, default=SEEDS)
    p.add_argument("--grid", type=int, nargs=4, default=list(GRID), metavar=("G1", "G2", "G3", "G4"))
    p.add_argument("--out", type=str, default=OUT_DIR)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    steps = int(args.steps)
    runs = int(args.runs)
    seeds = int(args.seeds)
    grid = tuple(int(x) for x in args.grid)
    out_dir = str(args.out)

    # Re-apply CI clamp after parsing (so CLI can't blow up smoke runs)
    if os.getenv("RG_CI", "0") == "1":
        steps = min(steps, 200)
        runs = min(runs, 2000)
        seeds = min(seeds, 1)
        print(f"[CI] trimming workload → steps={steps}, runs={runs}, seeds={seeds}, grid={grid}")

    result = run_demo(steps=steps, runs=runs, seeds=seeds, grid=grid, out_dir=out_dir)
    print("[ringing-demo] wrote:", Path(out_dir) / "ringing_demo_summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
