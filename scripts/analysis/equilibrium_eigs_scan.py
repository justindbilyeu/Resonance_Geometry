#!/usr/bin/env python3
"""Scan equilibria and eigenvalues for the driven pendulum RTP model."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

plt.rcParams["svg.fonttype"] = "none"


@dataclass
class EquilibriumResult:
    alpha: float
    branch_id: int
    phi_eq: float
    eigenvalues: Tuple[complex, complex]

    @property
    def stable(self) -> bool:
        return np.max(np.real(self.eigenvalues)) < 0.0


def equilibrium_residual(phi: float, alpha: float, w0: float, K0: float) -> float:
    """Equilibrium condition residual for the RTP model."""
    return (w0 ** 2) * phi - K0 * np.sin(alpha * phi)


def find_equilibria(
    alpha: float,
    w0: float,
    K0: float,
    grid_points: int = 400,
    dedupe_tol: float = 1e-6,
) -> List[float]:
    """Find equilibrium solutions for a given alpha via sign-change bracketing."""

    phi_limit = max(2.0, 1.5 * K0 / (w0 ** 2) + 1.0) if w0 != 0 else 10.0
    phi_grid = np.linspace(-phi_limit, phi_limit, grid_points)
    residuals = equilibrium_residual(phi_grid, alpha, w0, K0)

    roots: List[float] = []

    def register_root(phi_value: float) -> None:
        phi_value = float(phi_value)
        for existing in roots:
            if abs(existing - phi_value) < dedupe_tol:
                return
        roots.append(phi_value)

    for phi_value, residual in zip(phi_grid, residuals):
        if abs(residual) < 1e-10:
            register_root(phi_value)

    for x0, x1, f0, f1 in zip(phi_grid[:-1], phi_grid[1:], residuals[:-1], residuals[1:]):
        if f0 * f1 > 0:
            continue
        if abs(f0) < 1e-12 and abs(f1) < 1e-12:
            continue
        bracket = (x0, x1)
        try:
            root = optimize.root_scalar(
                equilibrium_residual,
                args=(alpha, w0, K0),
                bracket=bracket,
                method="brentq",
            )
        except ValueError:
            continue
        if root.converged:
            register_root(root.root)

    roots.sort()
    return roots


def assign_branch_ids(
    equilibria: Sequence[float],
    prev_mapping: Dict[int, float],
    next_branch_id: int,
    match_tol: float = 0.2,
) -> Tuple[List[Tuple[int, float]], int]:
    """Match equilibria to persistent branch identifiers."""

    if not equilibria:
        return [], next_branch_id

    if not prev_mapping:
        assigned = []
        for phi in equilibria:
            assigned.append((next_branch_id, phi))
            next_branch_id += 1
        return assigned, next_branch_id

    remaining_prev = prev_mapping.copy()
    assigned: List[Tuple[int, float]] = []

    for phi in equilibria:
        if remaining_prev:
            best_branch = min(
                remaining_prev.items(), key=lambda item: abs(item[1] - phi)
            )
            branch_id, prev_phi = best_branch
            if abs(prev_phi - phi) <= match_tol:
                assigned.append((branch_id, phi))
                remaining_prev.pop(branch_id)
                continue
        assigned.append((next_branch_id, phi))
        next_branch_id += 1

    return assigned, next_branch_id


def compute_eigenvalues(
    phi_eq: float,
    alpha: float,
    w0: float,
    gamma: float,
    K0: float,
) -> Tuple[complex, complex]:
    """Return eigenvalues of the Jacobian at the equilibrium."""

    stiffness = (w0 ** 2) - K0 * alpha * np.cos(alpha * phi_eq)
    jacobian = np.array([[0.0, 1.0], [-stiffness, -gamma]])
    eigvals = np.linalg.eigvals(jacobian)
    eigvals_sorted = tuple(sorted(eigvals, key=lambda val: np.real(val), reverse=True))
    return eigvals_sorted  # type: ignore[return-value]


def build_alpha_grid(alpha_min: float, alpha_max: float, alpha_step: float) -> np.ndarray:
    if alpha_step <= 0:
        raise ValueError("alpha_step must be positive")
    count = int(round((alpha_max - alpha_min) / alpha_step))
    if count < 0:
        raise ValueError("alpha_max must be >= alpha_min")
    grid = alpha_min + alpha_step * np.arange(count + 1)
    return np.round(grid, 10)


def run_scan(
    w0: float,
    gamma: float,
    K0: float,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    rtp_alpha_hint: float,
    results_dir: Path,
    figures_dir: Path,
) -> None:
    alpha_grid = build_alpha_grid(alpha_min, alpha_max, alpha_step)

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prev_mapping: Dict[int, float] = {}
    next_branch_id = 0

    records: List[EquilibriumResult] = []
    max_real_by_alpha: List[Tuple[float, float]] = []

    branch_traces: Dict[int, Dict[str, List[float]]] = {}

    for alpha in alpha_grid:
        equilibria = find_equilibria(alpha, w0, K0)
        branch_assignments, next_branch_id = assign_branch_ids(
            equilibria, prev_mapping, next_branch_id
        )
        current_mapping: Dict[int, float] = {}

        eigen_real_parts: List[float] = []

        for branch_id, phi_eq in branch_assignments:
            eigvals = compute_eigenvalues(phi_eq, alpha, w0, gamma, K0)
            records.append(EquilibriumResult(alpha, branch_id, phi_eq, eigvals))
            current_mapping[branch_id] = phi_eq
            branch_trace = branch_traces.setdefault(
                branch_id, {"alpha": [], "phi": [], "stable": []}
            )
            branch_trace["alpha"].append(alpha)
            branch_trace["phi"].append(phi_eq)
            branch_trace["stable"].append(
                bool(np.max(np.real(eigvals)) < -1e-9)
            )
            eigen_real_parts.extend(np.real(eigvals))

        max_real_by_alpha.append((float(alpha), float(np.max(eigen_real_parts)) if eigen_real_parts else float("nan")))
        prev_mapping = current_mapping

    summary = build_summary(
        records,
        max_real_by_alpha,
        rtp_alpha_hint=rtp_alpha_hint,
        parameters={
            "w0": w0,
            "gamma": gamma,
            "K0": K0,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_step": alpha_step,
        },
    )

    write_csv(records, results_dir / "alpha_scan_eigs.csv")
    write_summary(summary, results_dir / "alpha_scan_summary.json")

    plot_eigen_real(max_real_by_alpha, rtp_alpha_hint, summary, figures_dir)
    plot_eigen_imag(records, alpha_grid, figures_dir)
    plot_equilibria(branch_traces, rtp_alpha_hint, summary, figures_dir)


def build_summary(
    records: Sequence[EquilibriumResult],
    max_real_by_alpha: Sequence[Tuple[float, float]],
    *,
    rtp_alpha_hint: float,
    parameters: Dict[str, float],
) -> Dict[str, object]:
    overall_max = max((val for _, val in max_real_by_alpha), default=float("nan"))
    idx_max = next(
        (i for i, (_, val) in enumerate(max_real_by_alpha) if val == overall_max),
        0,
    )
    alpha_at_max = max_real_by_alpha[idx_max][0] if max_real_by_alpha else float("nan")
    alpha_critical = (parameters["w0"] ** 2) / parameters["K0"] if parameters["K0"] != 0 else float("inf")

    return {
        "parameters": parameters,
        "max_real_eig_by_alpha": [
            {"alpha": alpha, "max_real_eig": val} for alpha, val in max_real_by_alpha
        ],
        "rtp_alpha_hint": rtp_alpha_hint,
        "alpha_critical_w0_over_K0": alpha_critical,
        "overall_max_real_eig": {
            "value": overall_max,
            "alpha": alpha_at_max,
        },
        "record_count": len(records),
    }


def write_csv(records: Sequence[EquilibriumResult], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "alpha",
                "branch_id",
                "phi_eq",
                "eig1_real",
                "eig1_imag",
                "eig2_real",
                "eig2_imag",
                "stable_flag",
            ]
        )
        for record in records:
            eig1, eig2 = record.eigenvalues
            writer.writerow(
                [
                    f"{record.alpha:.6f}",
                    record.branch_id,
                    f"{record.phi_eq:.10f}",
                    f"{np.real(eig1):.10f}",
                    f"{np.imag(eig1):.10f}",
                    f"{np.real(eig2):.10f}",
                    f"{np.imag(eig2):.10f}",
                    int(record.stable),
                ]
            )


def write_summary(summary: Dict[str, object], path: Path) -> None:
    with path.open("w") as handle:
        json.dump(summary, handle, indent=2)


def plot_eigen_real(
    max_real_by_alpha: Sequence[Tuple[float, float]],
    rtp_alpha_hint: float,
    summary: Dict[str, object],
    figures_dir: Path,
) -> None:
    alphas = np.array([item[0] for item in max_real_by_alpha])
    values = np.array([item[1] for item in max_real_by_alpha])

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, values, label="max Re(λ)")
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.axvline(rtp_alpha_hint, color="red", linestyle=":", linewidth=1.0, label="α_RTP")
    alpha_critical = summary.get("alpha_critical_w0_over_K0")
    if isinstance(alpha_critical, (int, float)) and np.isfinite(alpha_critical):
        plt.axvline(
            alpha_critical,
            color="purple",
            linestyle="--",
            linewidth=1.0,
            label="α_c = ω₀²/K₀",
        )
    plt.xlabel("α")
    plt.ylabel("max Re(λ)")
    plt.title("Max real eigenvalue vs α")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    output_path = figures_dir / "eigenvalue_real_vs_alpha.svg"
    plt.savefig(output_path)
    plt.close()


def plot_eigen_imag(
    records: Sequence[EquilibriumResult],
    alpha_grid: Sequence[float],
    figures_dir: Path,
) -> None:
    alpha_to_imag: Dict[float, float] = {alpha: 0.0 for alpha in alpha_grid}
    for record in records:
        imag_parts = [abs(np.imag(val)) for val in record.eigenvalues]
        current = alpha_to_imag.get(record.alpha, 0.0)
        alpha_to_imag[record.alpha] = max(current, max(imag_parts))

    alphas = np.array(sorted(alpha_to_imag.keys()))
    values = np.array([alpha_to_imag[a] for a in alphas])

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, values, color="tab:green")
    plt.xlabel("α")
    plt.ylabel("max |Im(λ)|")
    plt.title("Imaginary part magnitude vs α")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    output_path = figures_dir / "eigenvalue_imag_vs_alpha.svg"
    plt.savefig(output_path)
    plt.close()


def plot_equilibria(
    branch_traces: Dict[int, Dict[str, List[float]]],
    rtp_alpha_hint: float,
    summary: Dict[str, object],
    figures_dir: Path,
) -> None:
    plt.figure(figsize=(6, 4))

    stable_label_used = False
    unstable_label_used = False

    for branch_id, data in sorted(branch_traces.items()):
        alphas = np.array(data["alpha"])
        phis = np.array(data["phi"])
        stable_mask = np.array(data["stable"], dtype=bool)

        sort_idx = np.argsort(alphas)
        alphas = alphas[sort_idx]
        phis = phis[sort_idx]
        stable_mask = stable_mask[sort_idx]

        plt.plot(alphas, phis, linewidth=1.2, label=f"Branch {branch_id}")
        if np.any(stable_mask):
            label = "Stable" if not stable_label_used else None
            plt.scatter(
                alphas[stable_mask],
                phis[stable_mask],
                color="tab:blue",
                s=25,
                label=label,
            )
            stable_label_used = True
        if np.any(~stable_mask):
            label = "Unstable" if not unstable_label_used else None
            plt.scatter(
                alphas[~stable_mask],
                phis[~stable_mask],
                color="tab:orange",
                marker="x",
                s=35,
                label=label,
            )
            unstable_label_used = True

    plt.axvline(rtp_alpha_hint, color="red", linestyle=":", linewidth=1.0)
    alpha_critical = summary.get("alpha_critical_w0_over_K0")
    if isinstance(alpha_critical, (int, float)) and np.isfinite(alpha_critical):
        plt.axvline(alpha_critical, color="purple", linestyle="--", linewidth=1.0)

    plt.xlabel("α")
    plt.ylabel("φ_eq")
    plt.title("Equilibria vs α")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    output_path = figures_dir / "equilibrium_vs_alpha.svg"
    plt.savefig(output_path)
    plt.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan equilibria and eigenvalues for the RTP model.",
    )
    parser.add_argument("--w0", type=float, default=1.0, help="Natural frequency (ω₀)")
    parser.add_argument("--gamma", type=float, default=0.08, help="Damping coefficient γ")
    parser.add_argument("--K0", type=float, default=0.8, help="Drive amplitude K₀")
    parser.add_argument("--alpha-min", type=float, default=0.25, dest="alpha_min")
    parser.add_argument("--alpha-max", type=float, default=0.45, dest="alpha_max")
    parser.add_argument("--alpha-step", type=float, default=0.002, dest="alpha_step")
    parser.add_argument(
        "--rtp-alpha",
        type=float,
        default=0.35,
        dest="rtp_alpha",
        help="Reference RTP alpha for plotting.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/equilibrium"),
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/assets/figures"),
        help="Directory for figure outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_scan(
        w0=args.w0,
        gamma=args.gamma,
        K0=args.K0,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        rtp_alpha_hint=args.rtp_alpha,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
