#!/usr/bin/env python3
"""Plot the eigenvalue scan curve from a persisted summary JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _load_summary(path: Path) -> tuple[Sequence[float], Sequence[float]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if {"alpha_grid", "max_real"} - payload.keys():
        raise ValueError("Summary file missing required keys {alpha_grid, max_real}")
    alphas = payload["alpha_grid"]
    max_real = [float("nan") if v is None else float(v) for v in payload["max_real"]]
    return alphas, max_real


def plot_curve(alphas: Sequence[float], max_real: Sequence[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, max_real, lw=2)
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("max Re(λ) across equilibria")
    ax.set_title("Eigenvalue scan vs alpha (K0=1.2, γ=0.08, ω0²=1)")
    y = np.asarray(max_real, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size:
        padding = 0.05
        ax.set_ylim(finite.min() - padding, finite.max() + padding)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/analysis/eigs_scan_summary.json"),
        help="Path to the JSON summary produced by scripts/equilibrium_analysis.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/eigenvalue_real_vs_alpha.png"),
        help="Destination for the generated PNG plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas, max_real = _load_summary(args.summary)
    plot_curve(alphas, max_real, args.out)


if __name__ == "__main__":
    main()
