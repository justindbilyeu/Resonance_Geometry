#!/usr/bin/env python3
# scripts/run_fluency_sweep.py
from __future__ import annotations

import inspect
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.fluency_velocity import (
    coherence_series,
    fluency_velocity,
    relaxation_velocity,
    summary_metrics,
)

# Try project simulator
HAS_SIM = False
try:
    from experiments.gp_ringing_demo import simulate_coupled

    SIG = inspect.signature(simulate_coupled)
    SUPPORTED = set(SIG.parameters.keys())
    HAS_SIM = True
except Exception:
    SUPPORTED = set()


def surrogate(alpha=0.1, beta=0.2, K0=0.2, tau=20.0, steps=800, seed=0, N=8):
    """Damped, driven coupled oscillators with noise."""

    rng = np.random.default_rng(seed)
    dt = 0.05
    omega = 1.0 / max(tau, 1e-6)
    zeta = beta
    A = K0
    Omega = max(alpha, 1e-3)

    x = np.zeros((steps, N))
    v = np.zeros(N)
    s = np.zeros(N)
    for t in range(steps):
        f = (
            A * np.sin(Omega * t * dt)
            - 2 * zeta * omega * v
            - (omega**2) * s
            + 0.02 * rng.standard_normal(N)
        )
        v += f * dt
        s += v * dt
        x[t] = s
    return np.arange(steps) * dt, x


def run_point(alpha, beta, K0, tau, steps=800, seed=42, perturb_at=400):
    if HAS_SIM:
        kwargs = {"alpha": alpha, "beta": beta, "K0": K0, "tau": tau, "steps": steps, "seed": seed}
        kwargs = {k: v for k, v in kwargs.items() if k in SUPPORTED}
        try:
            t, states = simulate_coupled(**kwargs)
        except Exception:
            t, states = surrogate(alpha, beta, K0, tau, steps, seed)
    else:
        t, states = surrogate(alpha, beta, K0, tau, steps, seed)

    # mid-run perturbation: small kick to all nodes
    if states.ndim == 2 and states.shape[0] > perturb_at:
        states[perturb_at:] += 0.1

    phi = coherence_series(states)
    vf = fluency_velocity(phi, smooth=True)
    vrel = relaxation_velocity(phi, t0=perturb_at, eq_window=min(100, len(phi) // 5))
    summ = summary_metrics(phi, vf)

    out = dict(alpha=alpha, beta=beta, K0=K0, tau=tau, v_relax=vrel, **summ)
    return out


def main():
    alphas = [0.05, 0.1, 0.2]
    betas = [0.02, 0.05, 0.2, 0.4]
    K0s = [0.0, 0.1, 0.3]
    taus = [10.0, 20.0]
    grid = list(product(alphas, betas, K0s, taus))

    outdir = Path("results/fluency_sweep")
    outdir.mkdir(parents=True, exist_ok=True)
    grid_path = outdir / "grid.jsonl"
    summ_path = outdir / "summary.json"
    meta_path = outdir / "meta.json"
    md_path = Path("docs/reports/fluency_velocity.md")
    fig_path = Path("docs/assets/figures/fluency_map.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    if grid_path.exists():
        grid_path.unlink()

    rows = []
    for (a, b, k, tau_val) in grid:
        row = run_point(a, b, k, tau_val, steps=800, seed=42, perturb_at=400)
        rows.append(row)
        with open(grid_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Aggregate
    def frac_ab(a, b):
        sel = [r for r in rows if r["alpha"] == a and r["beta"] == b]
        if not sel:
            return 0.0
        # proxy: high |vf| → “agile”; low v_relax → “slow recovery”
        return float(np.mean([abs(r["vf_p95"]) for r in sel]))

    # Simple heatmap α×β of vf_p95
    A = sorted(set([r["alpha"] for r in rows]))
    B = sorted(set([r["beta"] for r in rows]))
    M = np.zeros((len(B), len(A)))
    for i, b in enumerate(B):
        for j, a in enumerate(A):
            M[i, j] = frac_ab(a, b)

    plt.imshow(
        M,
        origin="lower",
        extent=[min(A), max(A), min(B), max(B)],
        aspect="auto",
    )
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("Fluency (|v_f| p95) across α–β")
    plt.colorbar(label="|v_f| p95")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    # Summary JSON
    by_param = {}
    for key in ["alpha", "beta", "K0", "tau"]:
        groups = {}
        vals = sorted(set([r[key] for r in rows]))
        for v in vals:
            sel = [r for r in rows if r[key] == v]
            groups[v] = {
                "n": len(sel),
                "vf_p95_mean": float(np.mean([s["vf_p95"] for s in sel])),
                "v_relax_mean": float(np.mean([s["v_relax"] for s in sel])),
            }
        by_param[key] = groups

    summary = {
        "total": len(rows),
        "vf_p95_global": float(np.mean([r["vf_p95"] for r in rows])),
        "v_relax_global": float(np.mean([r["v_relax"] for r in rows])),
        "by_param": by_param,
    }
    json.dump(summary, open(summ_path, "w"), indent=2)
    json.dump(
        {
            "has_simulator": HAS_SIM,
            "supported_params": sorted(list(SUPPORTED)),
        },
        open(meta_path, "w"),
        indent=2,
    )

    # Markdown report
    with open(md_path, "w") as f:
        f.write("# Fluency Velocity Sweep\n\n")
        f.write(
            "**Metric:** v_f = dΦ/dt (Savitzky–Golay smoothed), relaxation rate v_relax post-perturbation.\n\n"
        )
        f.write(
            "**Grid:** α ∈ %s, β ∈ %s, K₀ ∈ %s, τ ∈ %s.\n\n"
            % (
                A,
                B,
                sorted(set([r["K0"] for r in rows])),
                sorted(set([r["tau"] for r in rows])),
            )
        )
        f.write("**Figure:** ![Fluency map](../assets/figures/fluency_map.png)\n\n")
        best = sorted(rows, key=lambda r: r["vf_p95"], reverse=True)[:10]
        f.write("## Top-10 by |v_f| p95\n\n")
        f.write("| alpha | beta | K0 | tau | vf_p95 | v_relax |\n|------:|-----:|---:|----:|-------:|--------:|\n")
        for r in best:
            f.write(
                f"| {r['alpha']:.3f} | {r['beta']:.3f} | {r['K0']:.2f} | {r['tau']:.1f} | {r['vf_p95']:.4f} | {r['v_relax']:.4f} |\n"
            )
    print("Fluency sweep complete.")


if __name__ == "__main__":
    main()
