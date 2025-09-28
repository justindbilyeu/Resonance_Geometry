"""CLI entry point for topological accessibility coverage experiments."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from resonance_geometry.state_vector import extract_state_vector


DEFAULT_CONFIG = Path("config/topo_test.yaml")
DEFAULT_OUTPUT = Path("outputs/topo/coverage.npz")
DEFAULT_SAMPLES = Path("data/topo/samples.npz")


@dataclass
class ExperimentConfig:
    """Container for configuration values used by the experiment."""

    lambda_range: tuple[float, float]
    beta_range: tuple[float, float]
    A_range: tuple[float, float]
    runs: int
    schedule_length: int
    coupling_dim: int
    mi_length: int
    grid_min: np.ndarray
    grid_max: np.ndarray
    grid_resolution: np.ndarray
    random_seed: int
    drift: float
    g_noise_scale: float
    mi_noise_scale: float

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ExperimentConfig":
        sampling = raw.get("sampling", {})
        grid = raw.get("grid", {})
        stub = raw.get("stub_dynamics", {})

        return cls(
            lambda_range=_to_range(raw.get("lambda_range", [0.0, 1.0])),
            beta_range=_to_range(raw.get("beta_range", [0.0, 1.0])),
            A_range=_to_range(raw.get("A_range", [0.0, 1.0])),
            runs=int(sampling.get("runs", 1)),
            schedule_length=int(sampling.get("schedule_length", 1)),
            coupling_dim=int(sampling.get("coupling_dim", 2)),
            mi_length=int(sampling.get("mi_length", 1)),
            grid_min=_to_array(grid.get("min", [0.0] * 6)),
            grid_max=_to_array(grid.get("max", [1.0] * 6)),
            grid_resolution=_to_array(grid.get("resolution", [4] * 6)).astype(int),
            random_seed=int(raw.get("random_seed", 0)),
            drift=float(stub.get("drift", 0.9)),
            g_noise_scale=float(stub.get("g_noise_scale", 0.2)),
            mi_noise_scale=float(stub.get("mi_noise_scale", 0.1)),
        )


def _to_range(values: Iterable[float]) -> tuple[float, float]:
    lo, hi = list(values)[:2]
    return float(lo), float(hi)


def _to_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Topological accessibility coverage experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    map_parser = subparsers.add_parser(
        "map-accessible",
        help="Generate a 6D coverage map by sampling GP evolutions.",
    )
    map_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML configuration file.",
    )
    map_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override for reproducibility.",
    )
    map_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate a minimal dataset quickly for smoke testing.",
    )
    map_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON summary of the run instead of plain text.",
    )
    map_parser.set_defaults(func=run_map_accessible)

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def load_config(path: Path) -> ExperimentConfig:
    raw = _load_simple_yaml(path)
    return ExperimentConfig.from_dict(raw)


def _load_simple_yaml(path: Path) -> dict[str, object]:
    """Parse a small subset of YAML sufficient for the experiment config."""

    root: dict[str, object] = {}
    stack: list[tuple[dict[str, object], int]] = [(root, -1)]

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.split("#", 1)[0].rstrip()
            if not stripped:
                continue

            indent = len(raw_line) - len(raw_line.lstrip(" "))

            while stack and indent <= stack[-1][1] and len(stack) > 1:
                stack.pop()

            current = stack[-1][0]

            if stripped.endswith(":"):
                key = stripped[:-1].strip()
                new_dict: dict[str, object] = {}
                current[key] = new_dict
                stack.append((new_dict, indent))
                continue

            if ":" not in stripped:
                raise ValueError(f"Invalid line in config: {raw_line!r}")

            key, value_str = stripped.split(":", 1)
            key = key.strip()
            value = _parse_value(value_str.strip())
            current[key] = value

    return root


def _parse_value(token: str) -> object:
    if not token:
        return ""

    if token.startswith("[") and token.endswith("]"):
        return ast.literal_eval(token)

    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if any(ch in token for ch in ".eE"):
            return float(token)
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


def run_map_accessible(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    seed = args.seed if args.seed is not None else config.random_seed
    rng = np.random.default_rng(seed)

    runs = config.runs
    schedule_length = config.schedule_length
    mi_length = config.mi_length

    if args.dry_run:
        runs = min(runs, 4)
        schedule_length = min(schedule_length, 3)
        mi_length = min(mi_length, 8)

    state_vectors: list[np.ndarray] = []
    schedules: list[dict[str, list[float]]] = []

    for _ in range(runs):
        lambda_schedule = rng.uniform(*config.lambda_range, size=schedule_length)
        beta_schedule = rng.uniform(*config.beta_range, size=schedule_length)
        A_schedule = rng.uniform(*config.A_range, size=schedule_length)

        g_matrix = _simulate_coupling_matrix(
            rng,
            config.coupling_dim,
            schedule_length,
            config.drift,
            config.g_noise_scale,
        )
        mi_timeseries = _simulate_mi_series(
            rng,
            mi_length,
            config.drift,
            config.mi_noise_scale,
        )

        schedules.append(
            {
                "lambda": lambda_schedule.tolist(),
                "beta": beta_schedule.tolist(),
                "A": A_schedule.tolist(),
            }
        )

        state_vec = extract_state_vector(
            lambda_schedule[-1],
            beta_schedule[-1],
            A_schedule[-1],
            g_matrix,
            mi_timeseries,
        )
        state_vectors.append(state_vec)

    state_array = np.vstack(state_vectors) if state_vectors else np.empty((0, 6))

    coverage = _compute_coverage(
        state_array,
        config.grid_min,
        config.grid_max,
        config.grid_resolution,
    )

    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SAMPLES.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        DEFAULT_SAMPLES,
        state_vectors=state_array,
        schedules=np.array(schedules, dtype=object),
        seed=seed,
        dry_run=bool(args.dry_run),
        grid_min=config.grid_min,
        grid_max=config.grid_max,
        grid_resolution=config.grid_resolution,
    )

    np.savez(DEFAULT_OUTPUT, coverage=coverage.astype(np.uint8))

    summary = {
        "runs": int(runs),
        "seed": int(seed),
        "dry_run": bool(args.dry_run),
        "samples": int(state_array.shape[0]),
        "occupied_cells": int(np.count_nonzero(coverage)),
    }

    if args.json:
        print(json.dumps(summary))
    else:
        print(
            "Generated {samples} samples across {runs} runs (occupied cells: {occupied_cells}).".format(
                **summary
            )
        )

    return 0


def _simulate_coupling_matrix(
    rng: np.random.Generator,
    dim: int,
    steps: int,
    drift: float,
    noise: float,
) -> np.ndarray:
    matrix = rng.normal(scale=noise, size=(dim, dim))
    for _ in range(max(steps - 1, 0)):
        matrix = drift * matrix + rng.normal(scale=noise, size=(dim, dim))
    return matrix


def _simulate_mi_series(
    rng: np.random.Generator,
    length: int,
    drift: float,
    noise: float,
) -> np.ndarray:
    if length <= 0:
        return np.empty(0)
    series = rng.normal(scale=noise, size=length)
    for idx in range(1, length):
        series[idx] = drift * series[idx - 1] + rng.normal(scale=noise)
    return series


def _compute_coverage(
    state_vectors: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    if not len(state_vectors):
        return np.zeros(resolution, dtype=bool)

    spans = np.maximum(maxs - mins, 1e-9)
    scaled = (state_vectors - mins) / spans
    scaled = np.clip(scaled, 0.0, 0.999999)
    indices = (scaled * resolution).astype(int)

    coverage = np.zeros(resolution, dtype=bool)
    for index in indices:
        coverage[tuple(index)] = True
    return coverage


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        raise ValueError("No command specified")
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
