"""Tiny smoke test for the optimized spin-foam Monte Carlo driver."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from spin_foam_mc_optimized import optimized_spin_foam_mc


DEFAULT_STEPS = 5_000
DEFAULT_SIZE = 16
DEFAULT_RUNS = 4
DEFAULT_SEED: Optional[int] = 7


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--json", action="store_true", help="Emit the summary as JSON instead of text"
    )
    return parser.parse_args(argv)


def format_summary(summary: dict[str, object]) -> str:
    lines = [
        "Spin-foam optimized MC smoke summary:",
        f"  lattice size : {summary['size']}x{summary['size']}",
        f"  steps/run    : {summary['steps']}",
        f"  runs         : {summary['runs']}",
        f"  mean amp     : {summary['mean_amplitude']:.6f}",
        f"  mean energy  : {summary['mean_energy']:.6f}",
        f"  acceptance   : {summary['acceptance']:.3f}",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    summary = optimized_spin_foam_mc(
        steps=args.steps, size=args.size, runs=args.runs, seed=args.seed
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
