#!/usr/bin/env python3
"""Equilibrium + eigenvalue scan for RG playground."""
import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import fsolve

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class Params:
    w0_sq: float = 1.0     # ω0^2
    gamma: float = 0.08    # damping
    K0: float = 1.2        # drive (bumped to reach oscillatory regime)
    alpha: float = 0.35    # nonlinearity (swept)
    # equilibrium condition: w0_sq * φ_eq = K0 * sin(alpha * φ_eq)


def residual(phi: float, a: float, p: Params) -> float:
    return p.w0_sq * phi - p.K0 * np.sin(a * phi)


def solve_equilibria(a: float, p: Params, span=(-20, 20), n=49, tol=1e-8) -> List[float]:
    guesses = np.linspace(span[0], span[1], n)
    sols = []
    for g in guesses:
        try:
            sol = fsolve(lambda x: residual(x, a, p), g, xtol=1e-12, maxfev=1000)
            val = float(sol[0])
            # accept only near-zeros
            if abs(residual(val, a, p)) < 1e-6:
                if not any(abs(val - s) < tol for s in sols):
                    sols.append(val)
        except Exception:
            pass
    sols.sort()
    return sols


def jacobian(phi_eq: float, a: float, p: Params):
    """Linearization around (phi_eq, 0) for state [phi, dphi]."""
    A = np.array([[0.0, 1.0],
                  [-(p.w0_sq - p.K0 * a * np.cos(a * phi_eq)), -p.gamma]])
    return A


def sweep_alpha(
    a_min: float = 0.25,
    a_max: float = 0.55,
    steps: int = 61,
    csv_path: Union[Path, str] = "docs/analysis/eigs_scan_alpha.csv",
    json_path: Union[Path, str] = "docs/analysis/eigs_scan_summary.json",
    png_path: Union[Path, str] | None = None,
    svg_path: Union[Path, str] | None = None,
) -> dict:
    """Run the alpha sweep and persist CSV/JSON artifacts."""
    p = Params()
    alphas = np.linspace(a_min, a_max, steps)

    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path is not None:
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
    if svg_path is not None:
        svg_path = Path(svg_path)
        svg_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    max_real: List[float] = []
    any_eq: List[bool] = []
    for a in alphas:
        eqs = solve_equilibria(a, p)
        any_eq.append(len(eqs) > 0)
        if not eqs:
            max_real.append(float("nan"))
            continue
        eig_reals = []
        for phi_eq in eqs:
            A = jacobian(phi_eq, a, p)
            ev = eigvals(A)
            eig_reals.append(np.max(np.real(ev)))
            rows.append(
                {
                    "alpha": a,
                    "phi_eq": phi_eq,
                    "eig_real_max": float(np.max(np.real(ev))),
                    "eig_imag_mean": float(np.mean(np.imag(ev))),
                }
            )
        max_real.append(float(np.max(eig_reals)))

    # Write CSV of branch points
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["alpha", "phi_eq", "eig_real_max", "eig_imag_mean"]
        )
        writer.writeheader()
        writer.writerows(rows)

    if png_path is not None or svg_path is not None:
        import matplotlib.pyplot as plt  # Local import to avoid hard dependency when unused

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, max_real, lw=2)
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("alpha")
        ax.set_ylabel("max Re(λ) across equilibria")
        ax.set_title("Eigenvalue scan vs alpha (K0=1.2, γ=0.08, ω0²=1)")
        fig.tight_layout()
        if png_path is not None:
            fig.savefig(png_path, dpi=160)
        if svg_path is not None:
            fig.savefig(svg_path, format='svg')
        plt.close(fig)

    summary = {
        "alpha_grid": [float(x) for x in alphas],
        "max_real": [(None if np.isnan(x) else float(x)) for x in max_real],
        "equilibria_found": any_eq,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file.")
    parser.add_argument("--alpha-start", type=float, default=None, help="Alpha sweep start value.")
    parser.add_argument("--alpha-stop", type=float, default=None, help="Alpha sweep stop value.")
    parser.add_argument("--alpha-step", type=float, default=None, help="Alpha step size.")
    parser.add_argument("--alpha-steps", type=int, default=None, help="Number of alpha grid steps.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/analysis"),
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Path to write the CSV branch table (overrides --out-dir).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Path to write the JSON summary (overrides --out-dir).",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Optional path to write a generated PNG figure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Optional numerical seed (reserved for future stochastic extensions).",
    )
    parser.add_argument("--export-csv", action="store_true", help="Export CSV data.")
    parser.add_argument("--export-json", action="store_true", help="Export JSON summary.")
    parser.add_argument("--plot-svg", action="store_true", help="Generate SVG plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config if provided
    config = {}
    if args.config is not None:
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Determine alpha parameters (CLI overrides config)
    alpha_start = args.alpha_start if args.alpha_start is not None else config.get("alpha", {}).get("start", 0.25)
    alpha_stop = args.alpha_stop if args.alpha_stop is not None else config.get("alpha", {}).get("stop", 0.55)
    alpha_step = args.alpha_step if args.alpha_step is not None else config.get("alpha", {}).get("step", None)
    alpha_steps = args.alpha_steps

    # Calculate steps from step size if provided
    if alpha_step is not None and alpha_steps is None:
        alpha_steps = int((alpha_stop - alpha_start) / alpha_step) + 1
    elif alpha_steps is None:
        alpha_steps = 61  # default

    # Determine seed
    seed = int(os.environ.get("RG_SEED", args.seed if args.seed is not None else config.get("seed", 1337)))
    np.random.default_rng(seed)

    # Determine output paths
    out_dir = args.out_dir
    csv_path = args.out_csv if args.out_csv is not None else (out_dir / "eigs_scan_alpha.csv" if args.export_csv else None)
    json_path = args.out_json if args.out_json is not None else (out_dir / "eigs_scan_summary.json" if args.export_json else None)
    png_path = args.out_png

    # Derive SVG path from CSV name if plot_svg is set
    svg_path = None
    if args.plot_svg:
        if csv_path is not None:
            # Use the CSV basename to create corresponding SVG name
            csv_stem = csv_path.stem  # e.g., "eigs_scan_alpha" or "eigs_scan_alpha_narrow"
            svg_path = csv_path.parent / "figures" / f"{csv_stem.replace('eigs_scan_', 'eigenvalue_real_vs_')}.svg"
        else:
            svg_path = out_dir / "figures" / "eigenvalue_real_vs_alpha.svg"

    # Fallback to old defaults if no flags set
    if csv_path is None and json_path is None and png_path is None and svg_path is None:
        csv_path = out_dir / "eigs_scan_alpha.csv"
        json_path = out_dir / "eigs_scan_summary.json"

    sweep_alpha(
        a_min=alpha_start,
        a_max=alpha_stop,
        steps=alpha_steps,
        csv_path=csv_path if csv_path else out_dir / "eigs_scan_alpha.csv",
        json_path=json_path if json_path else out_dir / "eigs_scan_summary.json",
        png_path=png_path,
        svg_path=svg_path,
    )


if __name__ == "__main__":
    main()
