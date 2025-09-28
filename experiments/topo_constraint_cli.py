"""Command-line interface for topological constraint experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class StrategyResult:
    strategy: str
    time_multiplier: float
    distance: float
    hit: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "strategy": self.strategy,
            "time_multiplier": self.time_multiplier,
            "distance": self.distance,
            "hit": self.hit,
        }


def annealing(cell: np.ndarray, grid_spacing: float, time_multiplier: float) -> float:
    """Stub for an annealing-style search distance."""
    return _pseudo_distance(cell, grid_spacing, time_multiplier, phase=0.25)


def bang_bang(cell: np.ndarray, grid_spacing: float, time_multiplier: float) -> float:
    """Stub for a bang-bang control search distance."""
    return _pseudo_distance(cell, grid_spacing, time_multiplier, phase=0.5)


def noise_injection(cell: np.ndarray, grid_spacing: float, time_multiplier: float) -> float:
    """Stub for noisy forcing distance."""
    return _pseudo_distance(cell, grid_spacing, time_multiplier, phase=0.75)


def gradient_ascent(cell: np.ndarray, grid_spacing: float, time_multiplier: float) -> float:
    """Stub for gradient ascent forcing distance."""
    return _pseudo_distance(cell, grid_spacing, time_multiplier, phase=1.0)


STRATEGY_FUNCTIONS = {
    "annealing": annealing,
    "bang_bang": bang_bang,
    "noise_injection": noise_injection,
    "gradient_ascent": gradient_ascent,
}


def _pseudo_distance(
    cell: np.ndarray, grid_spacing: float, time_multiplier: float, *, phase: float
) -> float:
    """Deterministic synthetic distance used by the forcing stubs."""
    # The distance is bounded near the classification threshold so that some
    # strategies succeed for a subset of cells. The time multiplier shrinks the
    # distance, modelling the notion that more time allows the forcing routine to
    # converge closer to the target.
    base = float(np.sum(cell) + np.linalg.norm(cell))
    oscillation = abs(math.sin(base * (1.0 + phase)))
    normalized = 0.05 + 0.1 * oscillation
    return grid_spacing * normalized / max(time_multiplier, 1e-8)


def load_candidate_cells(path: Path) -> Tuple[np.ndarray, float]:
    if not path.exists():
        raise FileNotFoundError(f"Coverage file not found: {path}")

    data = np.load(path)
    if "candidate_cells" not in data or "grid_spacing" not in data:
        raise KeyError("coverage.npz must contain 'candidate_cells' and 'grid_spacing'")

    candidate_cells = np.asarray(data["candidate_cells"], dtype=float)
    grid_spacing = float(data["grid_spacing"].item())
    return candidate_cells, grid_spacing


def classify_cells(
    cells: Sequence[np.ndarray],
    grid_spacing: float,
    strategies: Iterable[str],
    time_budgets: Sequence[float],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    hit_threshold = 0.1 * grid_spacing
    baseline_budget = min(time_budgets)
    max_budget = max(time_budgets)

    per_cell: List[Dict[str, object]] = []
    counts = {"REACHABLE": 0, "RARE_BUT_REACHABLE": 0, "FORBIDDEN": 0}

    for index, cell in enumerate(cells):
        attempts: List[StrategyResult] = []
        earliest_hit_budget = None

        for time_multiplier in time_budgets:
            for strategy in strategies:
                func = STRATEGY_FUNCTIONS[strategy]
                distance = func(cell, grid_spacing, time_multiplier)
                hit = distance <= hit_threshold
                attempts.append(
                    StrategyResult(
                        strategy=strategy,
                        time_multiplier=time_multiplier,
                        distance=distance,
                        hit=hit,
                    )
                )
                if hit and earliest_hit_budget is None:
                    earliest_hit_budget = time_multiplier

        label = classify_label(earliest_hit_budget, baseline_budget, max_budget)
        counts[label] += 1
        per_cell.append(
            {
                "cell_index": index,
                "target": cell.tolist(),
                "label": label,
                "strategy_attempts": [attempt.to_dict() for attempt in attempts],
            }
        )

    return per_cell, counts


def classify_label(
    earliest_hit_budget: float | None, baseline_budget: float, max_budget: float
) -> str:
    if earliest_hit_budget is None:
        return "FORBIDDEN"
    if earliest_hit_budget <= baseline_budget:
        return "REACHABLE"
    if earliest_hit_budget <= min(4 * baseline_budget, max_budget):
        return "RARE_BUT_REACHABLE"
    return "FORBIDDEN"


def save_results(
    cells: List[Dict[str, object]],
    counts: Dict[str, int],
    output_path: Path,
    *,
    time_budgets: Sequence[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "total_cells": sum(counts.values()),
            "counts": counts,
            "time_budgets": list(time_budgets),
            "strategies": list(STRATEGY_FUNCTIONS.keys()),
        },
        "cells": cells,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_force_probe(args: argparse.Namespace) -> None:
    coverage_path = Path(args.coverage)
    cells, grid_spacing = load_candidate_cells(coverage_path)

    if args.dry_run:
        # Reduce the workload for smoke tests while keeping deterministic
        # behaviour.
        cells = cells[: min(len(cells), args.max_cells)]

    strategies = list(STRATEGY_FUNCTIONS.keys())
    time_budgets = tuple(sorted(set(float(b) for b in args.time_budget)))

    cell_results, counts = classify_cells(cells, grid_spacing, strategies, time_budgets)
    save_results(cell_results, counts, Path(args.output), time_budgets=time_budgets)

    if args.verbose:
        print(json.dumps({"summary": {"counts": counts}}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    force_parser = subparsers.add_parser(
        "force-probe", help="Run adversarial forcing over candidate cells."
    )
    force_parser.add_argument(
        "--coverage",
        type=str,
        default="outputs/topo/coverage.npz",
        help="Path to the coverage NPZ file.",
    )
    force_parser.add_argument(
        "--output",
        type=str,
        default="outputs/topo/forbidden.json",
        help="Destination for the JSON results.",
    )
    force_parser.add_argument(
        "--time-budget",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0],
        help="Time budget multipliers to evaluate.",
    )
    force_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Limit the number of cells for smoke testing.",
    )
    force_parser.add_argument(
        "--max-cells",
        type=int,
        default=3,
        help="Maximum cells to evaluate when running in dry-run mode.",
    )
    force_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a compact summary of the results to stdout.",
    )
    force_parser.set_defaults(func=run_force_probe)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
