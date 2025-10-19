#!/usr/bin/env python3
"""Run a lightweight ringing sweep, introspecting the real simulator when available."""
from __future__ import annotations
import json, inspect, sys
from pathlib import Path
from itertools import product
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional plotting in docs build only
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Try to import the real simulator
has_sim = False
try:
    from experiments.gp_ringing_demo import simulate_coupled
    sig = inspect.signature(simulate_coupled)
    supported = set(sig.parameters.keys())
    has_sim = True
except Exception:
    supported = set()

# Fallback surrogate
def surrogate(alpha, beta, eta, tau, K0, steps=600):
    dt = 0.1
    omega = 1.0/float(tau) if tau else 1.0
    zeta  = float(beta)
    A     = float(K0)
    Omega = float(alpha)
    x, v, t = 0.0, 0.0, 0.0
    out = np.empty(steps, dtype=float)
    for i in range(steps):
        out[i] = x
        f = A*np.sin(Omega*t) - 2*zeta*omega*v - (omega**2)*x + eta*np.random.randn()
        v += f*dt
        x += v*dt
        t += dt
    return out

# Small grid by default; allow expansion via env later
alphas = [0.05, 0.1]
betas  = [0.01, 0.05, 0.2, 0.4]
etas   = [0.0, 0.01]
taus   = [10.0]
K0s    = [0.1, 0.2]

out_dir = Path("results/ringing_sweep")
out_dir.mkdir(parents=True, exist_ok=True)
grid_jl = out_dir/"grid.jsonl"
summary_fp = out_dir/"summary.json"

from experiments.ringing_detector import detect_ringing

rows = []
if grid_jl.exists():
    grid_jl.unlink()

for a, b, e, tau, k0 in product(alphas, betas, etas, taus, K0s):
    params = {"alpha": a, "beta": b, "eta": e, "tau": tau, "K0": k0}
    if has_sim:
        kwargs = {k:v for k,v in params.items() if k in supported}
        try:
            t, states = simulate_coupled(steps=600, seed=42, **kwargs)
            series = states.mean(axis=1) if states.ndim > 1 else states
        except Exception:
            series = surrogate(**params)
    else:
        series = surrogate(**params)

    det = detect_ringing(series)
    row = {**params, **det}
    rows.append(row)
    with grid_jl.open("a") as f:
        f.write(json.dumps(row)+"\n")

# Summarize
total = len(rows)
ring_count = sum(r["ringing"] for r in rows)
by_beta = {}
for b in betas:
    gs = [r for r in rows if r["beta"] == b]
    frac = sum(x["ringing"] for x in gs)/len(gs) if gs else 0.0
    by_beta[b] = {"frac": frac, "count": len(gs)}

summary = {"total": total, "ringing_frac": (ring_count/total if total else 0.0), "by_beta": by_beta}
summary_fp.write_text(json.dumps(summary, indent=2))

# Optional heatmap
docs_fig = Path("docs/assets/figures")
docs_fig.mkdir(parents=True, exist_ok=True)
if plt:
    # alpha-beta fraction map
    a_vals = sorted(set(alphas))
    b_vals = sorted(set(betas))
    M = np.zeros((len(b_vals), len(a_vals)))
    for i,b in enumerate(b_vals):
        for j,a in enumerate(a_vals):
            gs = [r for r in rows if r["beta"]==b and r["alpha"]==a]
            M[i,j] = sum(x["ringing"] for x in gs)/len(gs) if gs else 0.0
    fig = plt.figure(figsize=(6,4))
    im = plt.imshow(M, origin="lower", extent=[min(a_vals), max(a_vals), min(b_vals), max(b_vals)], aspect="auto")
    plt.xlabel("alpha"); plt.ylabel("beta"); plt.title("Ringing fraction")
    plt.colorbar(im, label="fraction")
    plt.tight_layout()
    plt.savefig(docs_fig/"ringing_alpha_beta.png", dpi=160)
    plt.close(fig)
