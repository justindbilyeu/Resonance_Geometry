#!/usr/bin/env python3
# Equilibrium + eigenvalue scan for RG playground
import numpy as np
import csv, json
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from numpy.linalg import eigvals

@dataclass
class Params:
    w0_sq: float = 1.0     # ω0^2
    gamma: float = 0.08    # damping
    K0: float = 1.2        # drive (bumped to reach oscillatory regime)
    alpha: float = 0.35    # nonlinearity (swept)
    # equilibrium condition: w0_sq * φ_eq = K0 * sin(alpha * φ_eq)

def residual(phi, a: float, p: Params):
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
    # State [phi, dphi]; dynamics: d/dt[phi, dphi] = [dphi, ddphi]
    # linearization around (phi_eq, 0)
    # ddphi ≈ -gamma*dphi + (K0*a*cos(a*phi_eq) - w0_sq)*phi
    A = np.array([[0.0, 1.0],
                  [-(p.w0_sq - p.K0 * a * np.cos(a * phi_eq)), -p.gamma]])
    return A

def sweep_alpha(a_min=0.25, a_max=0.55, steps=61, outdir="docs/analysis"):
    p = Params()
    alphas = np.linspace(a_min, a_max, steps)
    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    # Track the dominant (max real) eigenvalue among all equilibria at each alpha
    max_real = []
    any_eq = []
    for a in alphas:
        eqs = solve_equilibria(a, p)
        any_eq.append(len(eqs) > 0)
        if not eqs:
            max_real.append(np.nan)
            continue
        eig_reals = []
        for phi_eq in eqs:
            A = jacobian(phi_eq, a, p)
            ev = eigvals(A)
            eig_reals.append(np.max(np.real(ev)))
            rows.append({"alpha": a, "phi_eq": phi_eq,
                         "eig_real_max": float(np.max(np.real(ev))),
                         "eig_imag_mean": float(np.mean(np.imag(ev)))})
        max_real.append(np.max(eig_reals))

    # Write CSV of branch points
    csv_path = outdir / "eigs_scan_alpha.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha","phi_eq","eig_real_max","eig_imag_mean"])
        w.writeheader()
        w.writerows(rows)

    # Plot max real(λ) vs alpha
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(alphas, max_real, lw=2)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("max Re(λ) across equilibria")
    ax.set_title("Eigenvalue scan vs alpha (K0=1.2, γ=0.08, ω0²=1)")
    fig.tight_layout()
    fig.savefig(outdir / "figures/eigenvalue_real_vs_alpha.png", dpi=160)
    plt.close(fig)

    # Quick JSON summary
    summary = {
        "alpha_grid":[float(x) for x in alphas],
        "max_real":[(None if np.isnan(x) else float(x)) for x in max_real]
    }
    with open(outdir / "eigs_scan_summary.json","w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    sweep_alpha()
