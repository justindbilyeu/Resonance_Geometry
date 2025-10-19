#!/usr/bin/env python3
from __future__ import annotations

import json
import os

def main() -> None:
    try:
        from experiments.gp_ringing_demo import simulate_coupled
        from experiments.fluency_velocity import fluency_velocity
    except Exception as exc:
        print("Dependencies missing:", exc)
        return

    os.makedirs("results/fluency_probe", exist_ok=True)
    t, X = simulate_coupled(steps=1200, seed=123)
    if getattr(X, "ndim", 1) == 1:
        X = X[:, None]
    out = fluency_velocity(X)
    with open("results/fluency_probe/summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("fluency:", out["vf_mean"], "±", out["vf_std"])


if __name__ == "__main__":
    main()
