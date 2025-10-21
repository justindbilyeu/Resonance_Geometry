#!/usr/bin/env python3
# Fractal dimension pipeline for RG playground

import numpy as np
from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr

@dataclass
class RGParams:
    w0: float = 1.0
    gamma: float = 0.08
    alpha: float = 0.45
    b: float = 0.25
    kappa: float = 1.0
    K0: float = 1.2       # ↑ drive to reach ringing
    tau: float = 20.0
    c: float = 0.4
    plastic: bool = True
    C0: float = 0.0
    phi0: float = 1.0     # ↑ initial amplitude to avoid flat runs
    dphi0: float = 0.0

def rhs(t, y, p: RGParams):
    phi, dphi, C = y
    tanh_phi = np.tanh(phi)
    sech2_phi = 1.0 / np.cosh(phi)**2
    g_sat = 2.0 * p.b * tanh_phi * sech2_phi
    dU = -p.kappa * np.sin(p.kappa * phi)
    ddphi = -p.gamma * dphi - (p.w0**2 - p.alpha * C) * phi - g_sat + p.K0 * dU
    dC = (-C + p.c * phi**2) / p.tau if p.plastic else 0.0
    return np.array([dphi, ddphi, dC], dtype=float)

def simulate(p: RGParams, t_end=400.0, dt=0.1) -> Dict[str, np.ndarray]:
    t_eval = np.arange(0.0, t_end + dt, dt)
    y0 = np.array([p.phi0, p.dphi0, p.C0], dtype=float)
    sol = solve_ivp(lambda t, y: rhs(t, y, p), [0, t_end], y0, t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return {"t": sol.t, "phi": sol.y[0], "dphi": sol.y[1], "C": sol.y[2]}

def ringing_detector(phi: np.ndarray, t: np.ndarray, tail_frac=0.4, min_peaks=4, rms_thresh=1e-5) -> Dict[str, float]:
    n = len(phi)
    start = int((1.0 - tail_frac) * n)
    tail = phi[start:]
    tail_z = tail - np.mean(tail)
    s = np.sign(tail_z)
    zero_cross = np.sum((s[1:] * s[:-1]) < 0)
    rms = np.sqrt(np.mean(tail_z**2))
    score = rms * (zero_cross / max(len(tail_z) - 1, 1)) if rms >= rms_thresh else 0.0
    ringing = 1 if (zero_cross >= min_peaks and rms >= rms_thresh) else 0
    return {"ringing": float(ringing), "ring_score": float(score), "rms": float(rms), "zc": float(zero_cross)}

def analyze_run(p: RGParams, sim: Dict[str, np.ndarray]) -> Dict:
    det = ringing_detector(sim["phi"], sim["t"])
    vf_p95 = float(np.percentile(np.abs(sim["dphi"]), 95))
    return {
        "params": vars(p),
        "metrics": {**det, "vf_p95": vf_p95},
        "final": {"C": float(sim["C"][-1])}
    }

def box_counting_dimension(phase_space: np.ndarray, epsilons: np.ndarray) -> float:
    rng = phase_space.max(0) - phase_space.min(0)
    if np.any(rng < 1e-8): return 0.0
    ps_norm = (phase_space - phase_space.min(0)) / rng
    N_boxes = []
    for eps in epsilons:
        bins = max(2, int(1.0/eps))
        hist, _ = np.histogramdd(ps_norm, bins=(bins, bins))
        N_boxes.append(np.count_nonzero(hist))
    log_eps = np.log(1.0 / epsilons)
    log_N = np.log(np.array(N_boxes) + 1e-10)
    D, _ = np.polyfit(log_eps, log_N, 1)
    return float(D)

def fractal_sweep(param_name: str, grid: np.ndarray, base: RGParams, t_end=400.0, dt=0.1, outdir="docs/analysis/fractal"):
    out = []
    epsilons = np.logspace(-1, -4, 10)
    outpath = Path(outdir); outpath.mkdir(parents=True, exist_ok=True)
    for val in grid:
        p = RGParams(**vars(base))
        setattr(p, param_name, float(val))
        sim = simulate(p, t_end=t_end, dt=dt)
        start = int(0.6 * len(sim["t"]))
        phsp = np.vstack([sim["phi"][start:], sim["dphi"][start:]]).T
        D_B = box_counting_dimension(phsp, epsilons)
        res = analyze_run(p, sim)
        res["metrics"]["param_value"] = float(val)
        res["metrics"]["fractal_dim"] = D_B
        out.append(res)

    json_path = outpath / f"fractal_sweep_{param_name}_plastic_{base.plastic}.json"
    with open(json_path, "w") as f: json.dump(out, f, indent=2)

    xs = [r["metrics"]["param_value"] for r in out]
    rings = [r["metrics"]["ring_score"] for r in out]
    dims = [r["metrics"]["fractal_dim"] for r in out]
    corr = pearsonr(rings, dims)[0] if len(set(rings))>1 else 0.0
    print(f"[corr] ring_score vs fractal_dim: {corr:.3f}")

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(xs, rings, "o-", lw=2, label="ring_score")
    ax1.set_xlabel(param_name); ax1.set_ylabel("ring_score", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(xs, dims, "s-", lw=2, color="C3", label="fractal_dim")
    ax2.set_ylabel("fractal_dim", color="C3")
    fig.suptitle(f"Fractal sweep vs {param_name} | plastic={base.plastic}")
    fig.tight_layout()
    fig.savefig(outpath / f"fractal_sweep_{param_name}_plot_plastic_{base.plastic}.png", dpi=160)
    plt.close(fig)
    return out

if __name__ == "__main__":
    base = RGParams()
    grid = np.linspace(0.25, 0.65, 17)
    fractal_sweep("alpha", grid, base)
    base.plastic = False
    fractal_sweep("alpha", grid, base)
