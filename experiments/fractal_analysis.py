#!/usr/bin/env python3
"""Fractal dimension analysis of phase boundaries."""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Any

def load_phase_data(phase_file: Path) -> np.ndarray:
    with open(phase_file) as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    points = data if isinstance(data, list) else data.get('points', [])
    if not points:
        raise ValueError(f"No points in {phase_file}")
    
    boundary_points = []
    for pt in points:
        alpha = pt.get('alpha')
        eta = pt.get('eta')
        is_ringing = pt.get('ringing', False)
        
        if alpha is not None and eta is not None:
            boundary_points.append([alpha, eta, 1 if is_ringing else 0])
    
    return np.array(boundary_points)

def extract_boundary(points: np.ndarray) -> np.ndarray:
    alpha_range = np.unique(points[:, 0])
    boundary = []
    
    for alpha in alpha_range:
        alpha_pts = points[points[:, 0] == alpha]
        if len(alpha_pts) < 2:
            continue
        alpha_pts = alpha_pts[alpha_pts[:, 1].argsort()]
        
        for i in range(len(alpha_pts) - 1):
            if alpha_pts[i, 2] != alpha_pts[i+1, 2]:
                boundary.append([alpha, (alpha_pts[i, 1] + alpha_pts[i+1, 1]) / 2])
    
    return np.array(boundary) if boundary else points[:, :2]

def box_counting(points: np.ndarray, epsilon: float) -> int:
    if len(points) == 0:
        return 0
    
    min_vals, max_vals = points.min(axis=0), points.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    normalized = (points - min_vals) / range_vals
    
    boxes = set()
    for point in normalized:
        boxes.add(tuple((point / epsilon).astype(int)))
    return len(boxes)

def compute_fractal_dimension(boundary: np.ndarray, n_scales: int = 20) -> Dict[str, Any]:
    epsilons = np.logspace(-2, -0.3, n_scales)
    counts = [box_counting(boundary, eps) for eps in epsilons]
    
    valid = [(e, c) for e, c in zip(epsilons, counts) if c > 0]
    if len(valid) < 3:
        return {"H": 0.0, "R2": 0.0, "log_inv_eps": [], "log_counts": [], 
                "fit": {"slope": 0.0, "intercept": 0.0}, "CI": [0.0, 0.0]}
    
    eps_valid, counts_valid = zip(*valid)
    x = np.log(1 / np.array(eps_valid))
    y = np.log(counts_valid)
    
    slope, intercept, r_value, _, std_err = stats.linregress(x, y)
    ci_half = 1.96 * std_err
    
    return {
        "H": float(slope),
        "R2": float(r_value**2),
        "log_inv_eps": x.tolist(),
        "log_counts": y.tolist(),
        "fit": {"slope": float(slope), "intercept": float(intercept)},
        "CI": [float(slope - ci_half), float(slope + ci_half)]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-file", type=Path, 
                       default=Path("results/phase/phase_sweep.json"))
    parser.add_argument("--n-scales", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("results/fractal"))
    args = parser.parse_args()
    
    print(f"Loading {args.phase_file}...")
    points = load_phase_data(args.phase_file)
    print(f"Loaded {len(points)} points")
    
    print("Extracting boundary...")
    boundary = extract_boundary(points)
    print(f"Found {len(boundary)} boundary points")
    
    if len(boundary) < 3:
        print("WARNING: Not enough boundary points")
        result = {"H": 0.0, "R2": 0.0, "warning": "Insufficient data"}
    else:
        print("Computing fractal dimension...")
        result = compute_fractal_dimension(boundary, args.n_scales)
        print(f"H = {result['H']:.3f}, R² = {result['R2']:.3f}")
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "fractal_summary.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    Path("docs/data/latest").mkdir(parents=True, exist_ok=True)
    with open("docs/data/latest/fractal.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved to {args.out_dir}/fractal_summary.json")
