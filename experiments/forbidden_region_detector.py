#!/usr/bin/env python3
import argparse, json, os, time, math, pathlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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


def largest_forbidden_component(forbidden_mask: np.ndarray) -> int:
    """Compute the size of the largest connected component in a 4D mask.

    The mask is assumed to be a boolean array where ``True`` indicates a
    forbidden cell. Connectivity is axis-aligned (Manhattan adjacency) and is
    evaluated with NetworkX to exercise the real library rather than the
    previous shim implementation.
    """

    if forbidden_mask.size == 0:
        return 0

    dims = forbidden_mask.shape
    graph = nx.Graph()

    for index in map(tuple, np.argwhere(forbidden_mask)):
        graph.add_node(index)
        for axis in range(len(dims)):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] < dims[axis]:
                neighbor_t = tuple(neighbor)
                if forbidden_mask[neighbor_t]:
                    graph.add_edge(index, neighbor_t)

    if graph.number_of_nodes() == 0:
        return 0

    return max(len(component) for component in nx.connected_components(graph))


def _run_forbidden_detector(
    grid,
    runs,
    steps,
    seeds,
    base_seed,
    out_dir,
    n,
    dt,
    *,
    show_progress,
    make_figures,
    update_progress,
):
    lam_min, lam_max = 0.0, 2.0
    beta_min, beta_max = 0.0, 2.0
    A_min, A_max = 0.0, 2.0
    gnorm_min, gnorm_max = 0.0, 5.0

    Nlam, Nbeta, NA, Ng = grid
    visited = np.zeros((Nlam, Nbeta, NA, Ng), dtype=bool)

    t0 = time.time()
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + s)
        for i in range(runs):
            if show_progress and (i + 1) % 1000 == 0:
                print(f"[random] {i+1}/{runs * seeds}")
            lam = rng.uniform(lam_min, lam_max)
            beta = rng.uniform(beta_min, beta_max)
            A = rng.uniform(A_min, A_max)
            x0 = rng.standard_normal(n)
            xT, _ = gp_toy_evolve(x0, lam, beta, A, steps=steps, dt=dt, rng=rng)
            gnorm = float(np.linalg.norm(xT))

            ijkl = cell_index_4d(
                [lam, beta, A, gnorm],
                [lam_min, beta_min, A_min, gnorm_min],
                [lam_max, beta_max, A_max, gnorm_max],
                [Nlam, Nbeta, NA, Ng],
            )
            visited[ijkl] = True

    total_cells = visited.size
    visited_cells = int(visited.sum())
    forbidden_cells = total_cells - visited_cells
    forbidden_pct = 100.0 * forbidden_cells / total_cells

    forbidden_mask = ~visited
    largest_cc = largest_forbidden_component(forbidden_mask)
    largest_cc_proxy = int(largest_cc)

    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "visited_4d.npy"), visited)

    summary = {
        "grid_res": int(Nlam),
        "total_cells": int(total_cells),
        "visited_cells": int(visited_cells),
        "forbidden_cells": int(forbidden_cells),
        "forbidden_pct": float(forbidden_pct),
        "largest_cc_proxy": int(largest_cc_proxy),
        "largest_forbidden_component": int(largest_cc),
        "random_runs": int(runs * seeds),
        "n": int(n),
        "steps": int(steps),
        "runtime_sec": round(time.time() - t0, 3),
        "decision_hint": "ESCALATE" if forbidden_pct > 5 else "TENTATIVE",
    }

    with open(os.path.join(out_dir, "forbidden_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if make_figures:
        figroot = "figures/forbidden_vX"
        ensure_dir(figroot)

        def save_proj(ax_i, ax_j, name):
            proj = project_pairs(visited, ax_i, ax_j)
            plt.figure(figsize=(4, 4))
            plt.imshow(~proj.T, origin="lower", aspect="auto", cmap="Reds")
            plt.title(f"Forbidden projection: {name}")
            plt.xlabel(["λ", "β", "A", "||g||"][ax_i])
            plt.ylabel(["λ", "β", "A", "||g||"][ax_j])
            plt.tight_layout()
            outpng = os.path.join(figroot, f"{name}.png")
            plt.savefig(outpng, dpi=120)
            plt.close()

        save_proj(0, 1, "forbidden_lam_beta")
        save_proj(0, 2, "forbidden_lam_A")
        save_proj(1, 2, "forbidden_beta_A")

    if update_progress:
        os.system("python tools/update_progress.py forbidden 100")

    return summary, visited


def minimal_forbidden_test(
    grid_res=8,
    n_random_runs=10_000,
    n=32,
    steps=200,
    dt=0.05,
    seed=42,
    out_dir="results/forbidden_v0",
    *,
    seeds=1,
):
    """Convenience wrapper used by tests for quick smoke runs."""

    grid = (grid_res, grid_res, grid_res, grid_res)
    summary, _ = _run_forbidden_detector(
        grid,
        n_random_runs,
        steps,
        seeds,
        seed,
        out_dir,
        n,
        dt,
        show_progress=False,
        make_figures=False,
        update_progress=False,
    )
    return summary


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

    summary, _ = _run_forbidden_detector(
        tuple(args.grid),
        args.runs,
        args.steps,
        args.seeds,
        args.seed,
        args.out,
        args.n,
        args.dt,
        show_progress=True,
        make_figures=True,
        update_progress=True,
    )

    print(f"[detector] wrote: {args.out}/forbidden_summary.json")
    print(f"[detector] steps={args.steps}, runs={args.runs}×{args.seeds}, grid={args.grid}")
    print(f"[detector] forbidden % = {summary['forbidden_pct']:.2f}")
    print(
        "[detector] largest forbidden component = "
        f"{summary['largest_forbidden_component']}"
    )
    print("[detector] DONE")

if __name__ == "__main__":
    main()
