#!/usr/bin/env python3
import argparse, json, os, pathlib, time
import numpy as np

from experiments.forbidden_region_detector import cell_index_4d, gp_toy_evolve

def sample_target_cells(visited, k=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    forbidden_idx = np.argwhere(~visited)
    if forbidden_idx.size == 0:
        return []
    k = min(k, len(forbidden_idx))
    sel = forbidden_idx[rng.choice(len(forbidden_idx), size=k, replace=False)]
    return [tuple(map(int, s)) for s in sel]

def attack_once(strategy, lam, beta, A, target_cell, mins, maxs, grid, steps=400, dt=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = 32
    x0 = rng.standard_normal(n)
    # naive schedules — keep simple and fast
    if strategy == "grad":
        # crude gradient-like nudging: slowly ramp lam toward edges
        lam_sched = np.linspace(lam, lam, steps)
        beta_sched = np.linspace(beta, beta, steps)
        A_sched = np.linspace(A, A, steps)
    elif strategy == "anneal":
        lam_sched = lam + 0.5*np.sin(np.linspace(0, 4*np.pi, steps))
        beta_sched = beta + 0.5*np.sin(np.linspace(0, 6*np.pi, steps))
        A_sched = A + 0.5*np.sin(np.linspace(0, 8*np.pi, steps))
    elif strategy == "bangbang":
        lam_sched = lam + 0.8*np.sign(np.sin(np.linspace(0, 8*np.pi, steps)))
        beta_sched = beta + 0.8*np.sign(np.sin(np.linspace(0, 10*np.pi, steps)))
        A_sched = A + 0.8*np.sign(np.sin(np.linspace(0, 12*np.pi, steps)))
    else:  # noise
        lam_sched = lam + 0.2*np.random.standard_normal(steps)
        beta_sched = beta + 0.2*np.random.standard_normal(steps)
        A_sched = A + 0.2*np.random.standard_normal(steps)

    x = x0
    for t in range(steps):
        x, L = gp_toy_evolve(x, lam_sched[t], beta_sched[t], A_sched[t], steps=1, dt=dt, rng=rng)
    gnorm = float(np.linalg.norm(x))
    idx = cell_index_4d(
        [lam, beta, A, gnorm],
        mins, maxs, grid
    )
    return idx == tuple(target_cell)

def main():
    ap = argparse.ArgumentParser(description="Adversarial forcing on putative forbidden cells.")
    ap.add_argument("--input", required=True, help="Path to forbidden_summary.json from detector.")
    ap.add_argument("--strategies", default="anneal,bangbang,noise,grad")
    ap.add_argument("--attempts", type=int, default=1000)
    ap.add_argument("--out", default="results/forbidden_adv")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    data = json.load(open(args.input))
    # load the corresponding visited array in same directory if present
    visited_path = os.path.join(os.path.dirname(args.input), "visited_4d.npy")
    if not os.path.exists(visited_path):
        raise FileNotFoundError(f"Cannot find visited_4d.npy next to {args.input}")
    visited = np.load(visited_path)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    # parameter bounds must mirror detector
    lam_min, lam_max = 0.0, 2.0
    beta_min, beta_max = 0.0, 2.0
    A_min,   A_max   = 0.0, 2.0
    gmin, gmax = 0.0, 5.0
    grid = [data["grid_res"]]*4
    mins = [lam_min, beta_min, A_min, gmin]
    maxs = [lam_max, beta_max, A_max, gmax]

    targets = sample_target_cells(visited, k=10, rng=rng)
    results = []
    for cell in targets:
        cell_res = {"cell": list(map(int, cell))}
        for strat in strategies:
            hits = 0
            atts = max(20, args.attempts // max(1, len(strategies)))  # spread budget
            for _ in range(atts):
                lam = rng.uniform(lam_min, lam_max)
                beta = rng.uniform(beta_min, beta_max)
                A = rng.uniform(A_min, A_max)
                if attack_once(strat, lam, beta, A, cell, mins, maxs, grid, steps=args.steps, rng=rng):
                    hits += 1
            cell_res[strat if strat!="grad" else "gradient_ascent"] = {"hits": int(hits), "attempts": int(atts)}
        results.append(cell_res)

    decision = "NOT_TRULY_FORBIDDEN" if any(
        (c.get("anneal", c.get("annealing", {"hits":0})) if False else None)  # placeholder
        for _ in [0]  # (keep structure simple)
    ) else ("NOT_TRULY_FORBIDDEN" if any(
        r[k]["hits"] > 0 for r in results for k in r if isinstance(r[k], dict)
    ) else "POTENTIALLY_FORBIDDEN")

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    out = {
        "grid_res": data["grid_res"],
        "tested": len(targets),
        "cells": results,
        "decision_hint": decision
    }
    with open(os.path.join(args.out, "adversarial_report.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("[adversarial] summary:", out)

if __name__ == "__main__":
    main()
