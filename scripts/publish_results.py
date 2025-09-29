"""
Collect latest run artifacts and publish a normalized bundle for GitHub Pages.
Usage:
  python scripts/publish_results.py --run results/topo_test/run_YYYYMMDD_HHMM --out docs/data/latest
"""
import argparse
import json
from pathlib import Path


def load_json(p):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="docs/data/latest")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Expected inputs (gracefully optional)
    summary = load_json(run / "summary.json") or {}
    forb_points = load_json(run / "forbidden_points.json") or {"accessible": [], "forbidden": []}
    fractal = load_json(run / "fractal" / "fractal.json")
    curvature = load_json(run / "curvature" / "curvature.json")
    hyst = load_json(run / "hysteresis.json")
    ring = load_json("results/ringing_map.json")
    mf = load_json("results/multi_frequency_results.json")

    # Normalize fractal for front-end
    if fractal and "scales" in fractal and "counts" in fractal:
        import numpy as np

        eps = (1.0 / (np.array(fractal["scales"], dtype=float))).tolist()
        log_inv = [float(-1 * __import__("math").log(e)) for e in eps]  # log(1/ε)
        log_counts = [float(__import__("math").log(c)) for c in fractal["counts"]]
        fractal = {
            "H": float(fractal.get("H", 0)),
            "R2": float(fractal.get("r2", 0)),
            "CI": fractal.get("ci", [0, 0]),
            "log_inv_eps": log_inv,
            "log_counts": log_counts,
            "fit": {"slope": float(fractal.get("H", 0)), "intercept": 0.0},
        }

    # Write bundle
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "forbidden_points.json", "w") as f:
        json.dump(forb_points, f, indent=2)
    if fractal:
        (out / "fractal.json").write_text(json.dumps(fractal, indent=2))
    if curvature:
        (out / "curvature.json").write_text(json.dumps(curvature, indent=2))
    if hyst:
        (out / "hysteresis.json").write_text(json.dumps(hyst, indent=2))
    if ring:
        (out / "ringing_map.json").write_text(json.dumps(ring, indent=2))
    if mf:
        (out / "multifreq.json").write_text(json.dumps(mf, indent=2))

    print(f"Published dashboard bundle from {run} → {out}")


if __name__ == "__main__":
    main()
