#!/usr/bin/env python3
"""Forbidden-region detector utilities and CLI."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# --------------------------
# Toy GP evolve (as in your v0)
# --------------------------
def gp_toy_evolve(
    state: np.ndarray,
    lam: float,
    beta: float,
    A: float,
    *,
    steps: int = 200,
    dt: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the surrogate GP dynamics for ``steps`` iterations."""

    if rng is None:
        rng = np.random.default_rng()
    n = state.shape[0]
    # simple laplacian on ring
    L = -2 * np.eye(n)
    for i in range(n):
        L[i, (i - 1) % n] = 1
        L[i, (i + 1) % n] = 1

    x = state.copy()
    for _ in range(steps):
        I_meas = np.tanh(x)  # surrogate info flow
        dx = (
            I_meas
            - lam * x
            - beta * (L @ x)
            - A * (I_meas - np.tanh(x))
        )  # same as v0 toy
        x = x + dt * dx + 0.01 * rng.standard_normal(size=n)
    return x, L


def cell_index_4d(
    vals: Sequence[float],
    mins: Sequence[float],
    maxs: Sequence[float],
    grid: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Quantise ``vals`` into the 4D grid defined by ``mins``/``maxs``/``grid``."""

    idx = []
    for v, lo, hi, n in zip(vals, mins, maxs, grid):
        vclip = float(np.clip(v, lo, hi))
        t = (vclip - lo) / (hi - lo + 1e-12)
        k = int(np.floor(t * n))
        k = max(0, min(n - 1, k))
        idx.append(k)
    return tuple(idx)  # type: ignore[return-value]


def project_pairs(mask4d: np.ndarray, axis_i: int, axis_j: int) -> np.ndarray:
    """Project the 4D occupancy mask along two axes."""

    axes = tuple(k for k in range(4) if k not in (axis_i, axis_j))
    proj = mask4d.max(axis=axes)
    return proj


def ensure_dir(path: os.PathLike[str] | str) -> None:
    """Create ``path`` (and parents) if it does not exist."""

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def compute_largest_forbidden_component(visited: np.ndarray) -> int:
    """Return the size of the largest connected forbidden component."""

    grid_shape = tuple(int(s) for s in visited.shape)
    graph = nx.grid_graph(dim=grid_shape)
    # Remove nodes corresponding to visited cells so only forbidden nodes remain.
    to_remove = [node for node in list(graph.nodes()) if visited[node]]
    graph.remove_nodes_from(to_remove)

    largest = 0
    for component in nx.connected_components(graph):
        largest = max(largest, len(component))
    return largest


def _parameter_bounds() -> Tuple[Sequence[float], Sequence[float]]:
    mins = [0.0, 0.0, 0.0, 0.0]
    maxs = [2.0, 2.0, 2.0, 5.0]
    return mins, maxs


def run_random_exploration(
    grid_shape: Sequence[int],
    *,
    n_random_runs: int,
    steps: int,
    seeds: int,
    base_seed: int,
    n: int,
    dt: float,
    log_progress: bool = False,
) -> Tuple[np.ndarray, float]:
    """Perform random scans of the parameter grid returning the visited mask."""

    visited = np.zeros(tuple(int(g) for g in grid_shape), dtype=bool)
    mins, maxs = _parameter_bounds()
    t0 = time.time()

    total_runs = n_random_runs * max(seeds, 1)
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        for i in range(n_random_runs):
            if log_progress and (i + 1) % 1000 == 0:
                print(f"[random] {i + 1}/{total_runs}")
            lam = rng.uniform(mins[0], maxs[0])
            beta = rng.uniform(mins[1], maxs[1])
            A = rng.uniform(mins[2], maxs[2])
            x0 = rng.standard_normal(n)
            xT, _ = gp_toy_evolve(x0, lam, beta, A, steps=steps, dt=dt, rng=rng)
            gnorm = float(np.linalg.norm(xT))

            ijkl = cell_index_4d([lam, beta, A, gnorm], mins, maxs, grid_shape)
            visited[ijkl] = True

    runtime = time.time() - t0
    return visited, runtime


def build_summary(
    visited: np.ndarray,
    grid_shape: Sequence[int],
    *,
    n_random_runs: int,
    seeds: int,
    n: int,
    steps: int,
    runtime: float,
) -> dict:
    """Assemble detector summary statistics for downstream consumers."""

    total_cells = int(visited.size)
    visited_cells = int(visited.sum())
    forbidden_cells = total_cells - visited_cells
    forbidden_pct = 100.0 * forbidden_cells / total_cells if total_cells else 0.0
    largest_component = compute_largest_forbidden_component(visited)

    summary = {
        "grid_res": int(grid_shape[0]) if grid_shape else 0,
        "grid_shape": [int(s) for s in grid_shape],
        "total_cells": total_cells,
        "visited_cells": visited_cells,
        "forbidden_cells": forbidden_cells,
        "forbidden_pct": float(forbidden_pct),
        # Maintain the legacy key but populate it with the real statistic.
        "largest_cc_proxy": int(largest_component),
        "largest_forbidden_component": int(largest_component),
        "random_runs": int(n_random_runs * max(seeds, 1)),
        "n": int(n),
        "steps": int(steps),
        "runtime_sec": round(runtime, 3),
        "decision_hint": "ESCALATE" if forbidden_pct > 5 else "TENTATIVE",
    }
    return summary


def write_detector_outputs(
    out_dir: os.PathLike[str] | str,
    visited: np.ndarray,
    summary: dict,
    *,
    generate_figures: bool,
    update_progress: bool,
) -> dict:
    """Persist detector artefacts and return the enriched summary."""

    ensure_dir(out_dir)
    visited_path = os.path.join(out_dir, "visited_4d.npy")
    np.save(visited_path, visited)

    summary = dict(summary)
    summary["visited_path"] = visited_path
    summary_path = os.path.join(out_dir, "forbidden_summary.json")
    summary["summary_path"] = summary_path
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if generate_figures:
        ensure_dir("figures/forbidden_vX")  # generic sink
        figroot = "figures/forbidden_vX"

        labels = ["λ", "β", "A", "||g||"]

        def save_proj(ax_i: int, ax_j: int, name: str) -> None:
            proj = project_pairs(visited, ax_i, ax_j)  # True = visited
            plt.figure(figsize=(4, 4))
            plt.imshow(~proj.T, origin="lower", aspect="auto", cmap="Reds")  # red = forbidden
            plt.title(f"Forbidden projection: {name}")
            plt.xlabel(labels[ax_i])
            plt.ylabel(labels[ax_j])
            plt.tight_layout()
            outpng = os.path.join(figroot, f"{name}.png")
            plt.savefig(outpng, dpi=120)
            plt.close()

        save_proj(0, 1, "forbidden_lam_beta")
        save_proj(0, 2, "forbidden_lam_A")
        save_proj(1, 2, "forbidden_beta_A")

    if update_progress:
        os.system("python tools/update_progress.py forbidden 100")

    return summary


def minimal_forbidden_test(
    *,
    grid_res: int = 6,
    n_random_runs: int = 200,
    n: int = 8,
    steps: int = 100,
    dt: float = 0.05,
    seed: int = 123,
    out_dir: str = "results/forbidden_test",
    generate_figures: bool = False,
    update_progress: bool = False,
) -> dict:
    """Convenience wrapper used by the test-suite to exercise the detector."""

    grid_shape = tuple(int(grid_res) for _ in range(4))
    visited, runtime = run_random_exploration(
        grid_shape,
        n_random_runs=n_random_runs,
        steps=steps,
        seeds=1,
        base_seed=seed,
        n=n,
        dt=dt,
        log_progress=False,
    )
    summary = build_summary(
        visited,
        grid_shape,
        n_random_runs=n_random_runs,
        seeds=1,
        n=n,
        steps=steps,
        runtime=runtime,
    )
    return write_detector_outputs(
        out_dir,
        visited,
        summary,
        generate_figures=generate_figures,
        update_progress=update_progress,
    )


def main() -> None:
    """CLI entry point for the forbidden-region detector."""

    p = argparse.ArgumentParser(
        description="Forbidden-region minimal detector (now actually respects CLI flags)."
    )
    p.add_argument(
        "--grid",
        nargs=4,
        type=int,
        metavar=("N1", "N2", "N3", "N4"),
        default=[8, 8, 8, 8],
        help="Grid resolution per dim (λ, β, A, ||g||).",
    )
    p.add_argument("--runs", type=int, default=10000, help="Number of random starts.")
    p.add_argument("--steps", type=int, default=200, help="Integration steps per run.")
    p.add_argument("--seeds", type=int, default=1, help="Number of random seeds.")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    p.add_argument("--out", type=str, default="results/forbidden_v0", help="Output dir.")
    p.add_argument("--n", type=int, default=32, help="State dimension for toy system.")
    p.add_argument("--dt", type=float, default=0.05, help="Integrator dt.")
    args = p.parse_args()

    grid_shape = tuple(int(g) for g in args.grid)
    visited, runtime = run_random_exploration(
        grid_shape,
        n_random_runs=args.runs,
        steps=args.steps,
        seeds=args.seeds,
        base_seed=args.seed,
        n=args.n,
        dt=args.dt,
        log_progress=True,
    )
    summary = build_summary(
        visited,
        grid_shape,
        n_random_runs=args.runs,
        seeds=args.seeds,
        n=args.n,
        steps=args.steps,
        runtime=runtime,
    )
    summary = write_detector_outputs(
        args.out,
        visited,
        summary,
        generate_figures=True,
        update_progress=True,
    )

    print(f"[detector] wrote: {summary['summary_path']}")
    print(
        f"[detector] steps={args.steps}, runs={args.runs}×{args.seeds}, grid={list(grid_shape)}"
    )
    print(f"[detector] forbidden % = {summary['forbidden_pct']:.2f}")
    print("[detector] DONE")


if __name__ == "__main__":
    main()
