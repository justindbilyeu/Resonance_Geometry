#!/usr/bin/env python3
"""Generate phase trajectories for specified alpha values."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.resonance_geometry.hallucination.phase_dynamics import simulate_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.30, 0.40, 0.25, 0.35, 0.45, 0.50, 0.55],
        help="Alpha values to simulate (duplicates removed while preserving order).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/phase/traces"),
        help="Output directory for trajectory JSON files.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Seed for RNG.")
    parser.add_argument("--lam", type=float, default=1.0, help="Lambda parameter.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter.")
    parser.add_argument("--k", type=float, default=1.0, help="Spring constant parameter.")
    parser.add_argument("--beta", type=float, default=0.02, help="Quintic saturation coefficient.")
    parser.add_argument("--skew", type=float, default=0.12, help="Skew coupling coefficient.")
    parser.add_argument("--mi-window", type=int, default=30, dest="mi_window", help="Window size for MI estimate.")
    parser.add_argument("--mi-ema", type=float, default=0.1, dest="mi_ema", help="EMA factor for MI.")
    parser.add_argument("--eta", type=float, default=2.0, help="Eta parameter for drive strength.")
    return parser.parse_args()


def unique_preserve_order(values: Sequence[float]) -> Iterable[float]:
    seen: set[float] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        yield v


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    base_params = {
        "lambda": args.lam,
        "gamma": args.gamma,
        "k": args.k,
        "beta": args.beta,
        "skew": args.skew,
        "mu": 0.0,
        "mi_window": args.mi_window,
        "mi_ema": args.mi_ema,
        "omega_anchor": np.zeros(3),
        "eta": args.eta,
    }

    env_seed = os.environ.get("RG_SEED")
    seed = int(env_seed) if env_seed is not None else args.seed

    for alpha in unique_preserve_order(args.alphas):
        params = base_params.copy()
        params["alpha"] = float(alpha)
        traj = simulate_trajectory(params, seed=seed)
        payload = {
            "alpha": float(alpha),
            "t": traj["t"].tolist(),
            "S1": traj["norm"].tolist(),
        }
        target = outdir / f"traj_alpha_{alpha:.2f}.json"
        target.write_text(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
