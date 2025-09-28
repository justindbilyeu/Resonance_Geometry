import json
import os

import numpy as np

from experiments.forbidden_region_detector import gp_toy_evolve


# Target forbidden cells discovered by Task 1 and attempt to reach them.

def _bin_idx(vals, x):
    i = int(np.clip(np.searchsorted(vals, x, side="right") - 1, 0, len(vals) - 1))
    return i


def _mk_ranges(grid_res):
    lam_vals = np.linspace(0.1, 1.5, grid_res)
    beta_vals = np.linspace(0.0, 0.6, grid_res)
    A_vals = np.linspace(0.0, 1.2, grid_res)
    g_bins = np.linspace(0.0, 5.0, grid_res + 1)
    return lam_vals, beta_vals, A_vals, g_bins


def _strategy_gradient_ascent(target_cell, grid_res, attempts=20, evolve_kwargs=None):
    # crude coordinate hill-climb over (λ, β, A) to steer g_norm bin
    lam_vals, beta_vals, A_vals, g_bins = _mk_ranges(grid_res)
    successes = 0
    rng = np.random.default_rng(42)
    evolve_kwargs = evolve_kwargs or {}
    for _ in range(attempts):
        lam = rng.choice(lam_vals)
        beta = rng.choice(beta_vals)
        A = rng.choice(A_vals)
        best = None
        for _ in range(15):
            base = gp_toy_evolve(lam=lam, beta=beta, A=A, seed=rng.integers(1e9), **evolve_kwargs)
            gnorm = base["g_norm"]
            lbin = _bin_idx(g_bins, gnorm)
            cell = (
                _bin_idx(lam_vals, lam),
                _bin_idx(beta_vals, beta),
                _bin_idx(A_vals, A),
                lbin,
            )
            dist = sum(abs(ci - ti) for ci, ti in zip(cell, target_cell))
            if best is None or dist < best[0]:
                best = (dist, lam, beta, A)
            # small nudges
            lam += rng.normal(0, np.diff(lam_vals).mean() * 0.5)
            beta += rng.normal(0, np.diff(beta_vals).mean() * 0.5)
            A += rng.normal(0, np.diff(A_vals).mean() * 0.5)
        if best and best[0] == 0:
            successes += 1
    return successes, attempts


def _strategy_annealing(target_cell, grid_res, attempts=20, evolve_kwargs=None):
    lam_vals, beta_vals, A_vals, g_bins = _mk_ranges(grid_res)
    successes = 0
    rng = np.random.default_rng(43)
    evolve_kwargs = evolve_kwargs or {}
    for _ in range(attempts):
        lam = rng.choice(lam_vals)
        beta = rng.choice(beta_vals)
        A = rng.choice(A_vals)
        T = 1.0
        for _ in range(25):
            res = gp_toy_evolve(lam=lam, beta=beta, A=A, seed=rng.integers(1e9), **evolve_kwargs)
            gnorm = res["g_norm"]
            lbin = _bin_idx(g_bins, gnorm)
            cell = (
                _bin_idx(lam_vals, lam),
                _bin_idx(beta_vals, beta),
                _bin_idx(A_vals, A),
                lbin,
            )
            dist = sum(abs(ci - ti) for ci, ti in zip(cell, target_cell))
            if dist == 0:
                successes += 1
                break
            # propose jump
            lam2 = lam + rng.normal(0, np.diff(lam_vals).mean())
            beta2 = beta + rng.normal(0, np.diff(beta_vals).mean())
            A2 = A + rng.normal(0, np.diff(A_vals).mean())
            res2 = gp_toy_evolve(
                lam=lam2,
                beta=beta2,
                A=A2,
                seed=rng.integers(1e9),
                **evolve_kwargs,
            )
            gnorm2 = res2["g_norm"]
            lbin2 = _bin_idx(g_bins, gnorm2)
            cell2 = (
                _bin_idx(lam_vals, lam2),
                _bin_idx(beta_vals, beta2),
                _bin_idx(A_vals, A2),
                lbin2,
            )
            dist2 = sum(abs(ci - ti) for ci, ti in zip(cell2, target_cell))
            if dist2 <= dist or rng.random() < np.exp(-(dist2 - dist) / max(T, 1e-4)):
                lam, beta, A = lam2, beta2, A2
            T *= 0.9
    return successes, attempts


def _strategy_bang_bang(target_cell, grid_res, attempts=20, evolve_kwargs=None):
    lam_vals, beta_vals, A_vals, g_bins = _mk_ranges(grid_res)
    successes = 0
    rng = np.random.default_rng(44)
    evolve_kwargs = evolve_kwargs or {}
    for _ in range(attempts):
        lam_lo, lam_hi = lam_vals[0], lam_vals[-1]
        beta_lo, beta_hi = beta_vals[0], beta_vals[-1]
        A_lo, A_hi = A_vals[0], A_vals[-1]
        seq = [(lam_lo, beta_hi, A_lo), (lam_hi, beta_lo, A_hi)] * 12
        hit = False
        for lam, beta, A in seq:
            res = gp_toy_evolve(
                lam=lam,
                beta=beta,
                A=A,
                seed=rng.integers(1e9),
                **evolve_kwargs,
            )
            gnorm = res["g_norm"]
            lbin = _bin_idx(g_bins, gnorm)
            cell = (
                _bin_idx(lam_vals, lam),
                _bin_idx(beta_vals, beta),
                _bin_idx(A_vals, A),
                lbin,
            )
            if cell == tuple(target_cell):
                successes += 1
                hit = True
                break
        if not hit:
            pass
    return successes, attempts


def _strategy_noise_injection(target_cell, grid_res, attempts=20, evolve_kwargs=None):
    lam_vals, beta_vals, A_vals, g_bins = _mk_ranges(grid_res)
    successes = 0
    rng = np.random.default_rng(45)
    evolve_kwargs = evolve_kwargs or {}
    for _ in range(attempts):
        lam = rng.choice(lam_vals)
        beta = rng.choice(beta_vals)
        A = rng.choice(A_vals)
        for _ in range(25):
            lam += rng.normal(0, np.diff(lam_vals).mean())
            beta += rng.normal(0, np.diff(beta_vals).mean())
            A += rng.normal(0, np.diff(A_vals).mean())
            res = gp_toy_evolve(
                lam=lam,
                beta=beta,
                A=A,
                seed=rng.integers(1e9),
                **evolve_kwargs,
            )
            gnorm = res["g_norm"]
            lbin = _bin_idx(g_bins, gnorm)
            cell = (
                _bin_idx(lam_vals, lam),
                _bin_idx(beta_vals, beta),
                _bin_idx(A_vals, A),
                lbin,
            )
            if cell == tuple(target_cell):
                successes += 1
                break
    return successes, attempts


def adversarial_attack_pipeline(
    forbidden_summary_path="results/forbidden_v0/forbidden_summary.json",
    visited_path="results/forbidden_v0/visited_4d.npy",
    out_path="results/forbidden_v0/adversarial_report.json",
    max_forbidden_to_test=10,
    strategy_attempts=20,
    evolve_kwargs=None,
):
    with open(forbidden_summary_path, "r") as f:
        summ = json.load(f)
    visited = np.load(visited_path)
    grid_res = int(summ["grid_res"])
    if evolve_kwargs is None:
        evolve_kwargs = {
            "n": int(summ.get("n", 8)),
            "steps": int(summ.get("steps", 200)),
        }
    # collect forbidden cells
    coords = np.argwhere(~visited)
    rng = np.random.default_rng(123)
    rng.shuffle(coords)
    targets = coords[: min(max_forbidden_to_test, len(coords))]

    report = {"grid_res": grid_res, "tested": len(targets), "cells": []}
    for cell in targets:
        cell = cell.tolist()
        res_g, tot_g = _strategy_gradient_ascent(
            cell, grid_res, attempts=strategy_attempts, evolve_kwargs=evolve_kwargs
        )
        res_a, tot_a = _strategy_annealing(
            cell, grid_res, attempts=strategy_attempts, evolve_kwargs=evolve_kwargs
        )
        res_b, tot_b = _strategy_bang_bang(
            cell, grid_res, attempts=strategy_attempts, evolve_kwargs=evolve_kwargs
        )
        res_n, tot_n = _strategy_noise_injection(
            cell, grid_res, attempts=strategy_attempts, evolve_kwargs=evolve_kwargs
        )
        cell_report = {
            "cell": cell,
            "gradient_ascent": {"hits": res_g, "attempts": tot_g},
            "annealing": {"hits": res_a, "attempts": tot_a},
            "bang_bang": {"hits": res_b, "attempts": tot_b},
            "noise_injection": {"hits": res_n, "attempts": tot_n},
        }
        report["cells"].append(cell_report)

    # Decision: if ANY strategy hits a target → not truly forbidden
    any_hit = any(
        (
            c["gradient_ascent"]["hits"] > 0
            or c["annealing"]["hits"] > 0
            or c["bang_bang"]["hits"] > 0
            or c["noise_injection"]["hits"] > 0
        )
        for c in report["cells"]
    )
    report["decision_hint"] = (
        "NOT_TRULY_FORBIDDEN" if any_hit else "PERSISTS_AS_FORBIDDEN"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("[adversarial] summary:", report)
    return report


if __name__ == "__main__":
    adversarial_attack_pipeline()
