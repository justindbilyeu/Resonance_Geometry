#!/usr/bin/env python3
import argparse, json, os, time, math, pathlib
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Toy GP evolve (as in your v0)
# --------------------------
def gp_toy_evolve(state, lam, beta, A, steps=200, dt=0.05, rng=None):
    """
    Minimal surrogate dynamics:
    - state: (n,) vector
    - lam, beta, A: scalars
    """
    if rng is None:
        rng = np.random.default_rng()
    n = state.shape[0]
    # simple laplacian on ring
    L = -2*np.eye(n)
    for i in range(n):
        L[i,(i-1)%n] = 1
        L[i,(i+1)%n] = 1

    x = state.copy()
    for _ in range(steps):
        I_meas = np.tanh(x)  # surrogate info flow
        dx = (I_meas - lam*x - beta*(L@x) - A*(I_meas - np.tanh(x)))  # same as v0 toy
        x = x + dt*dx + 0.01*rng.standard_normal(size=n)
    return x, L

def cell_index_4d(vals, mins, maxs, grid):
    idx = []
    for v, lo, hi, n in zip(vals, mins, maxs, grid):
        vclip = np.clip(v, lo, hi)
        t = (vclip - lo) / (hi - lo + 1e-12)
        k = int(np.floor(t * n))
        k = max(0, min(n-1, k))
        idx.append(k)
    return tuple(idx)

def project_pairs(mask4d, axis_i, axis_j):
    # max-over other dims to show occupied/visited projection
    M = mask4d
    axes = tuple(k for k in range(4) if k not in (axis_i, axis_j))
    proj = M.max(axis=axes)
    return proj

def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def main():
    p = argparse.ArgumentParser(
        description="Forbidden-region minimal detector (now actually respects CLI flags)."
    )
    p.add_argument("--grid", nargs=4, type=int, metavar=("N1","N2","N3","N4"),
                   default=[8,8,8,8], help="Grid resolution per dim (λ, β, A, ||g||).")
    p.add_argument("--runs", type=int, default=10000, help="Number of random starts.")
    p.add_argument("--steps", type=int, default=200, help="Integration steps per run.")
    p.add_argument("--seeds", type=int, default=1, help="Number of random seeds.")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    p.add_argument("--out", type=str, default="results/forbidden_v0", help="Output dir.")
    p.add_argument("--n", type=int, default=32, help="State dimension for toy system.")
    p.add_argument("--dt", type=float, default=0.05, help="Integrator dt.")
    args = p.parse_args()

    # Parameter box (same ranges as v0)
    lam_min, lam_max = 0.0, 2.0
    beta_min, beta_max = 0.0, 2.0
    A_min,   A_max   = 0.0, 2.0
    gnorm_min, gnorm_max = 0.0, 5.0

    Nlam, Nbeta, NA, Ng = args.grid
    visited = np.zeros((Nlam, Nbeta, NA, Ng), dtype=bool)

    t0 = time.time()
    total_runs = args.runs * args.seeds
    for s in range(args.seeds):
        rng = np.random.default_rng(args.seed + s)
        for i in range(args.runs):
            if (i+1) % 1000 == 0:
                print(f"[random] {i+1}/{args.runs * args.seeds}")
            lam = rng.uniform(lam_min, lam_max)
            beta = rng.uniform(beta_min, beta_max)
            A    = rng.uniform(A_min,   A_max)
            x0 = rng.standard_normal(args.n)
            xT, L = gp_toy_evolve(x0, lam, beta, A, steps=args.steps, dt=args.dt, rng=rng)
            gnorm = float(np.linalg.norm(xT))

            ijkl = cell_index_4d(
                [lam, beta, A, gnorm],
                [lam_min, beta_min, A_min, gnorm_min],
                [lam_max, beta_max, A_max, gnorm_max],
                [Nlam, Nbeta, NA, Ng]
            )
            visited[ijkl] = True

    # Summaries
    total_cells = visited.size
    visited_cells = int(visited.sum())
    forbidden_cells = total_cells - visited_cells
    forbidden_pct = 100.0 * forbidden_cells / total_cells

    # crude proxy for largest connected component in forbidden set (2D slice heuristic)
    # (keep simple for now)
    largest_cc_proxy = int((~visited).astype(int).max())

    ensure_dir(args.out)
    # Save raw occupancy
    np.save(os.path.join(args.out, "visited_4d.npy"), visited)

    # Save summary
    summary = {
        "grid_res": int(Nlam),    # square grid assumed in 4 dims for brevity of label
        "total_cells": int(total_cells),
        "visited_cells": int(visited_cells),
        "forbidden_cells": int(forbidden_cells),
        "forbidden_pct": float(forbidden_pct),
        "largest_cc_proxy": int(largest_cc_proxy),
        "random_runs": int(args.runs * args.seeds),
        "n": int(args.n),
        "steps": int(args.steps),
        "runtime_sec": round(time.time() - t0, 3),
        "decision_hint": "ESCALATE" if forbidden_pct > 5 else "TENTATIVE"
    }
    with open(os.path.join(args.out, "forbidden_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Quick projections (λ–β, λ–A, β–A) with ||g|| maxed out
    ensure_dir("figures/forbidden_vX")  # generic sink
    figroot = "figures/forbidden_vX"

    def save_proj(ax_i, ax_j, name):
        proj = project_pairs(visited, ax_i, ax_j)  # True = visited
        plt.figure(figsize=(4,4))
        plt.imshow(~proj.T, origin="lower", aspect="auto", cmap="Reds")  # red = forbidden
        plt.title(f"Forbidden projection: {name}")
        plt.xlabel(["λ","β","A","||g||"][ax_i])
        plt.ylabel(["λ","β","A","||g||"][ax_j])
        plt.tight_layout()
        outpng = os.path.join(figroot, f"{name}.png")
        plt.savefig(outpng, dpi=120)
        plt.close()

    save_proj(0,1,"forbidden_lam_beta")
    save_proj(0,2,"forbidden_lam_A")
    save_proj(1,2,"forbidden_beta_A")

    print(f"[detector] wrote: {args.out}/forbidden_summary.json")
    print(f"[detector] steps={args.steps}, runs={args.runs}×{args.seeds}, grid={args.grid}")
    print(f"[detector] forbidden % = {forbidden_pct:.2f}")
    print("[detector] DONE")

if __name__ == "__main__":
    main()
