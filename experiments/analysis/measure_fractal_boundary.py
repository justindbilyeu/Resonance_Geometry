"""Estimate the fractal dimension of a forbidden-region boundary via box counting."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("outputs/topo/forbidden.json")
DEFAULT_OUTPUT = Path("outputs/topo/fractal_dim.json")
DEFAULT_FIGURE = Path("figures/topo/boxcount_fit.png")
N_BOOTSTRAP = 1_000
N_SCALES = 20
SEED = 2024
MIN_BOUNDARY_POINTS = 100


@dataclass
class BoxCountResult:
    slope: float
    intercept: float
    scales: np.ndarray
    counts: np.ndarray
    r2: float
    ci_low: float
    ci_high: float
    n_boundary: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of bootstrap resamples to draw",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Seed for the bootstrap RNG"
    )
    return parser.parse_args(argv)


def load_forbidden_map(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    shape = payload.get("shape")
    coords = payload.get("forbidden")
    if not isinstance(shape, list) or len(shape) != 6:
        raise ValueError("Forbidden map JSON must contain a 6D 'shape' list")
    if not isinstance(coords, list):
        raise ValueError("Forbidden map JSON must contain a 'forbidden' list of coordinates")

    array = np.zeros(shape, dtype=bool)
    for coord in coords:
        if not isinstance(coord, Iterable):
            raise ValueError("Forbidden coordinates must be iterable")
        index = tuple(int(c) for c in coord)
        array[index] = True
    return array


def extract_boundary_points(forbidden: np.ndarray) -> np.ndarray:
    if forbidden.ndim != 6:
        raise ValueError("Forbidden map must be 6-dimensional")

    boundary_mask = np.zeros_like(forbidden, dtype=bool)
    forbidden_indices = np.argwhere(forbidden)
    if forbidden_indices.size == 0:
        return forbidden_indices

    shape = forbidden.shape
    for idx in forbidden_indices:
        idx_tuple = tuple(idx)
        for axis in range(forbidden.ndim):
            for delta in (-1, 1):
                neighbor = idx.copy()
                neighbor[axis] += delta
                if neighbor[axis] < 0 or neighbor[axis] >= shape[axis]:
                    boundary_mask[idx_tuple] = True
                    break
                if not forbidden[tuple(neighbor)]:
                    boundary_mask[idx_tuple] = True
                    break
            if boundary_mask[idx_tuple]:
                break
    return np.argwhere(boundary_mask)


def _compute_box_counts(points: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if points.size == 0:
        raise ValueError("Cannot compute box counts with no boundary points")

    mins = points.min(axis=0)
    shifted = points - mins
    counts = []
    for scale in scales:
        bins = np.floor(shifted / scale).astype(int)
        unique = np.unique(bins, axis=0)
        counts.append(unique.shape[0])
    return np.asarray(counts, dtype=float)


def fit_box_counts(
    points: np.ndarray,
    n_scales: int = N_SCALES,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> BoxCountResult:
    if points.shape[0] < MIN_BOUNDARY_POINTS:
        raise RuntimeError(
            f"Insufficient boundary points for stable estimate: {points.shape[0]} < {MIN_BOUNDARY_POINTS}"
        )

    ranges = points.max(axis=0) - points.min(axis=0) + 1
    max_extent = ranges.max()
    min_extent = 1.0
    if max_extent <= min_extent:
        max_extent = min_extent + 1.0

    scales = np.logspace(math.log10(min_extent), math.log10(max_extent), n_scales)
    counts = _compute_box_counts(points, scales)

    log_counts = np.log(counts)
    log_inv_scale = np.log(1.0 / scales)
    slope, intercept = np.polyfit(log_inv_scale, log_counts, 1)
    pred = slope * log_inv_scale + intercept
    ss_res = np.sum((log_counts - pred) ** 2)
    ss_tot = np.sum((log_counts - log_counts.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(seed)
    boot_slopes = np.empty(n_bootstrap, dtype=float)
    n_points = points.shape[0]
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n_points, size=n_points)
        sample_points = points[sample_idx]
        sample_counts = _compute_box_counts(sample_points, scales)
        log_counts_bs = np.log(sample_counts)
        slope_bs, _ = np.polyfit(log_inv_scale, log_counts_bs, 1)
        boot_slopes[i] = slope_bs

    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])

    return BoxCountResult(
        slope=float(slope),
        intercept=float(intercept),
        scales=scales,
        counts=counts,
        r2=float(r2),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_boundary=int(points.shape[0]),
    )


def save_results(result: BoxCountResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality = "reliable" if result.r2 >= 0.90 else "unreliable"
    payload = {
        "H": result.slope,
        "ci": [result.ci_low, result.ci_high],
        "r2": result.r2,
        "n_boundary": result.n_boundary,
        "quality": quality,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_figure(result: BoxCountResult, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(result.scales, result.counts, color="tab:blue", label="Box counts")

    sorted_scales = np.linspace(result.scales.min(), result.scales.max(), 200)
    fit_counts = np.exp(result.intercept) * (1.0 / sorted_scales) ** result.slope
    ax.plot(sorted_scales, fit_counts, color="tab:orange", label=f"Fit H={result.slope:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Box size")
    ax.set_ylabel("Occupied boxes")
    ax.set_title("Boundary box-count fit")
    ax.legend()
    ax.text(
        0.05,
        0.95,
        f"R$^2$ = {result.r2:.3f}\n95% CI = [{result.ci_low:.3f}, {result.ci_high:.3f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    forbidden = load_forbidden_map(args.input)
    boundary_points = extract_boundary_points(forbidden)

    result = fit_box_counts(
        boundary_points.astype(float),
        n_scales=N_SCALES,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    save_results(result, args.output)
    save_figure(result, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
