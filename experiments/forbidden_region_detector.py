import json
import math
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Minimal GP toy evolution (self-contained) ----------
# We use a simple proxy: g is an n×n coupling matrix that evolves under a
# gradient flow driven by an "information drive" I (random but temporally
# correlated) plus regularization terms λ, β, A. This is a toy that matches
# our repo’s GP intuition without importing heavy internals.

def _laplacian_2d(n):
    L = np.zeros((n * n, n * n))
    idx = lambda i, j: i * n + j
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ii, jj = (i + di) % n, (j + dj) % n
                kk = idx(ii, jj)
                L[k, k] -= 1
                L[k, kk] += 1
    return L


def gp_toy_evolve(n=8, steps=200, seed=None, lam=0.5, beta=0.1, A=0.5, dt=0.05):
    rng = np.random.default_rng(seed)
    # state: g (n×n), I(t) (n×n)
    g = rng.normal(0, 0.1, size=(n, n))
    I = rng.normal(0, 1.0, size=(n, n))
    L = _laplacian_2d(n)
    gvec = g.reshape(-1)
    Ivec = I.reshape(-1)
    # simple colored noise driver for I
    rho = 0.98
    for t in range(steps):
        Ivec = rho * Ivec + math.sqrt(1 - rho**2) * rng.normal(0, 1.0, size=Ivec.shape)
        # toy "predicted" info from g (linear readout)
        Ipred = gvec.copy()
        grad = (
            -(Ivec)  # drives toward info
            + lam * gvec  # L2 regularization
            + beta * (L @ gvec)  # smoothness
            + A * (Ipred - Ivec)
        )  # alignment penalty
        gvec = gvec - dt * grad
    g = gvec.reshape(n, n)
    # emergent coordinates we’ll track
    g_norm = float(np.linalg.norm(g))
    smooth = float(gvec.T @ (L @ gvec))
    mi_range = float(np.max(Ivec) - np.min(Ivec))
    return {"g": g, "g_norm": g_norm, "smooth": smooth, "mi_range": mi_range}


# ---------- 4D grid + random exploration ----------
def minimal_forbidden_test(
    grid_res=8,
    n_random_runs=10_000,
    n=8,
    steps=200,
    seed=0,
    out_dir="results/forbidden_v0",
):
    """
    Quick 4D scan over (λ, β, A, ||g|| target-bin). We discretize the
    *emergent* ||g|| into grid_res bins to form a 4th axis so we can map
    visited cells from random exploration.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Parameter ranges (lightweight & defensible)
    lam_vals = np.linspace(0.1, 1.5, grid_res)
    beta_vals = np.linspace(0.0, 0.6, grid_res)
    A_vals = np.linspace(0.0, 1.2, grid_res)
    g_bins = np.linspace(0.0, 5.0, grid_res + 1)  # emergent ||g|| binning

    visited = np.zeros((grid_res, grid_res, grid_res, grid_res), dtype=bool)

    t0 = time.time()
    for run in range(n_random_runs):
        lam = rng.choice(lam_vals)
        beta = rng.choice(beta_vals)
        A = rng.choice(A_vals)
        res = gp_toy_evolve(
            n=n, steps=steps, seed=rng.integers(1e9), lam=lam, beta=beta, A=A
        )
        gnorm = res["g_norm"]

        i = int(np.clip(np.searchsorted(lam_vals, lam, side="right") - 1, 0, grid_res - 1))
        j = int(np.clip(np.searchsorted(beta_vals, beta, side="right") - 1, 0, grid_res - 1))
        k = int(np.clip(np.searchsorted(A_vals, A, side="right") - 1, 0, grid_res - 1))
        l = int(np.clip(np.searchsorted(g_bins, gnorm, side="right") - 1, 0, grid_res - 1))
        visited[i, j, k, l] = True

        if (run + 1) % 1000 == 0:
            print(f"[random] {run+1}/{n_random_runs}")

    total_cells = visited.size
    visited_count = int(visited.sum())
    forbidden_count = total_cells - visited_count
    forbidden_pct = 100.0 * forbidden_count / total_cells

    # Largest connected forbidden component (4D adjacency = Manhattan-1)
    # For speed, approximate by counting slices’ max area in 2D projections.
    # (Exact 4D CC is possible but heavier; MVP goes with projections.)

    def proj_largest_cc(mask2d):
        # naive BFS CC on 2D projection
        H, W = mask2d.shape
        seen = np.zeros_like(mask2d, dtype=bool)
        best = 0
        for y in range(H):
            for x in range(W):
                if mask2d[y, x] and not seen[y, x]:
                    # BFS
                    q = [(y, x)]
                    seen[y, x] = True
                    c = 1
                    while q:
                        yy, xx = q.pop()
                        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            y2, x2 = yy + dy, xx + dx
                            if (
                                0 <= y2 < H
                                and 0 <= x2 < W
                                and mask2d[y2, x2]
                                and not seen[y2, x2]
                            ):
                                seen[y2, x2] = True
                                q.append((y2, x2))
                                c += 1
                    best = max(best, c)
        return best

    forbidden = ~visited
    # project along each axis and take max CC as a heuristic
    cc_scores = []
    cc_scores.append(proj_largest_cc(forbidden.any(axis=(2, 3)).astype(int)))  # (i,j)
    cc_scores.append(proj_largest_cc(forbidden.any(axis=(1, 3)).astype(int)))  # (i,k)
    cc_scores.append(proj_largest_cc(forbidden.any(axis=(1, 2)).astype(int)))  # (i,l)
    cc_scores.append(proj_largest_cc(forbidden.any(axis=(0, 3)).astype(int)))  # (j,k)
    cc_scores.append(proj_largest_cc(forbidden.any(axis=(0, 2)).astype(int)))  # (j,l)
    cc_scores.append(proj_largest_cc(forbidden.any(axis=(0, 1)).astype(int)))  # (k,l)
    largest_cc_proxy = int(max(cc_scores))

    # Plots: simple 2D projections of forbidden space (% forbidden per slice)
    figdir = "figures/forbidden_v0"
    os.makedirs(figdir, exist_ok=True)

    def heat(name, arr2d, xticks, yticks, xlabel, ylabel):
        plt.figure(figsize=(5, 4))
        plt.imshow(100.0 * arr2d, origin="lower", aspect="auto", cmap="magma")
        plt.colorbar(label="% forbidden")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, f"{name}.png"))
        plt.close()

    # Percent forbidden per (λ,β)
    forb_lam_beta = forbidden.mean(axis=(2, 3))
    heat("forbidden_lam_beta", forb_lam_beta, lam_vals, beta_vals, "λ index", "β index")

    forb_lam_A = forbidden.mean(axis=(1, 3))
    heat("forbidden_lam_A", forb_lam_A, lam_vals, A_vals, "λ index", "A index")

    forb_beta_A = forbidden.mean(axis=(0, 3))
    heat("forbidden_beta_A", forb_beta_A, beta_vals, A_vals, "β index", "A index")

    summary = {
        "grid_res": grid_res,
        "total_cells": int(total_cells),
        "visited_cells": visited_count,
        "forbidden_cells": forbidden_count,
        "forbidden_pct": forbidden_pct,
        "largest_cc_proxy": largest_cc_proxy,
        "random_runs": n_random_runs,
        "n": n,
        "steps": steps,
        "runtime_sec": round(time.time() - t0, 3),
        "decision_hint": "ESCALATE" if forbidden_pct > 1.0 else "STAND_DOWN",
    }
    with open(os.path.join(out_dir, "forbidden_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.save(os.path.join(out_dir, "visited_4d.npy"), visited)
    print("[forbidden_v0] summary:", summary)
    return summary


if __name__ == "__main__":
    minimal_forbidden_test()
