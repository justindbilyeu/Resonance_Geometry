#!/usr/bin/env python3
"""Phase sweep for ringing boundary detection.

Sweeps over alpha (damping) and eta (plasticity) to identify where
systems transition from smooth to oscillatory (ringing) behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Import the tools we just created
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.stability import detect_boundary


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def run_rwp_dynamics(
    alpha: float,
    eta: float,
    n: int = 20,
    T: int = 200,
    dt: float = 0.05,
    seed: int = 42
) -> np.ndarray:
    """Run simplified RWP dynamics.
    
    Args:
        alpha: Damping parameter
        eta: Plasticity strength
        n: System size
        T: Time steps
        dt: Integration timestep
        seed: Random seed
        
    Returns:
        Trajectory of first node (for boundary detection)
    """
    rng = np.random.default_rng(seed)
    
    # Initialize state
    x = rng.standard_normal(n) * 0.1
    
    # Simple ring topology Laplacian
    L = -2 * np.eye(n)
    for i in range(n):
        L[i, (i-1) % n] = 1
        L[i, (i+1) % n] = 1
    
    # Coupling matrix (starts uniform, plasticity would adapt it)
    K = np.ones_like(L) * 0.1
    K[np.eye(n, dtype=bool)] = 0
    
    trajectory = np.zeros(T)
    
    for t in range(T):
        # RWP-like dynamics: damping + coupling + noise
        I_flow = np.tanh(x)  # Information flow proxy
        coupling_term = (K * L) @ I_flow
        
        dx = -alpha * x + coupling_term + 0.01 * rng.standard_normal(n)
        x = x + dt * dx
        
        # Simple plasticity update (Hebbian-like)
        if eta > 0:
            for i in range(n):
                for j in range(i+1, n):
                    if L[i, j] != 0:  # Only update actual connections
                        dK = eta * I_flow[i] * I_flow[j]
                        K[i, j] += dK * dt
                        K[j, i] = K[i, j]
        
        trajectory[t] = x[0]  # Track first node
    
    return trajectory


def analyze_point(
    alpha: float,
    eta: float,
    n: int,
    T: int,
    M: int,
    seed: int
) -> Dict:
    """Run multiple trials and detect ringing."""
    psd_peaks = []
    overshoots = []
    
    for trial in range(M):
        traj = run_rwp_dynamics(alpha, eta, n=n, T=T, seed=seed + trial)
        
        # Detect boundary using both methods
        psd_result = detect_boundary(traj, method="psd", fs=1.0/0.05)
        overshoot_result = detect_boundary(traj, method="overshoot")
        
        psd_peaks.append(psd_result["value"])
        overshoots.append(overshoot_result["value"])
    
    # Ringing criteria: high PSD peak AND significant overshoots
    avg_psd = float(np.mean(psd_peaks))
    avg_overshoot = float(np.mean(overshoots))
    
    # Conservative thresholds
    is_ringing = (avg_psd > 0.1) and (avg_overshoot > 0.05)
    
    return {
        "alpha": float(alpha),
        "eta": float(eta),
        "psd_peak": avg_psd,
        "overshoot": avg_overshoot,
        "ringing": bool(is_ringing)
    }


def run_phase_sweep(
    alphas: List[float],
    etas: List[float],
    n: int,
    T: int,
    M: int,
    seed: int,
    out_dir: str
) -> None:
    """Run full phase sweep over alpha x eta grid."""
    
    ensure_dir(out_dir)
    
    results = []
    total = len(alphas) * len(etas)
    count = 0
    
    print(f"Running phase sweep: {len(alphas)} alphas × {len(etas)} etas = {total} points")
    print(f"Each point: {M} trials, {T} timesteps, {n} nodes")
    
    for alpha in alphas:
        for eta in etas:
            count += 1
            print(f"[{count}/{total}] alpha={alpha:.3f}, eta={eta:.4f}...", end=" ")
            
            result = analyze_point(alpha, eta, n, T, M, seed)
            results.append(result)
            
            status = "RINGING" if result["ringing"] else "smooth"
            print(f"{status} (PSD={result['psd_peak']:.4f})")
    
    # Save results
    results_file = os.path.join(out_dir, "phase_sweep.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate phase map
    generate_phase_map(results, alphas, etas, out_dir)


def generate_phase_map(
    results: List[Dict],
    alphas: List[float],
    etas: List[float],
    out_dir: str
) -> None:
    """Generate phase boundary heatmap."""
    
    # Create grid
    grid = np.zeros((len(alphas), len(etas)))
    
    for result in results:
        i = alphas.index(result["alpha"])
        j = etas.index(result["eta"])
        grid[i, j] = 1.0 if result["ringing"] else 0.0
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(grid.T, origin="lower", aspect="auto", cmap="RdYlGn_r",
               extent=[alphas[0], alphas[-1], etas[0], etas[-1]])
    plt.colorbar(label="Ringing (1=yes, 0=no)")
    plt.xlabel("Alpha (damping)")
    plt.ylabel("Eta (plasticity)")
    plt.title("Phase Boundary: Ringing Region")
    
    out_file = os.path.join(out_dir, "phase_map.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Phase map saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase sweep for ringing boundary")
    parser.add_argument("--alphas", type=str, required=True,
                       help="Comma-separated alpha values (e.g., '0.1,0.3,0.6')")
    parser.add_argument("--etas", type=str, required=True,
                       help="Comma-separated eta values (e.g., '0.01,0.03,0.05')")
    parser.add_argument("--n", type=int, default=20,
                       help="System size (number of nodes)")
    parser.add_argument("--T", type=int, default=200,
                       help="Number of timesteps per trial")
    parser.add_argument("--M", type=int, default=10,
                       help="Number of trials per parameter point")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--out_dir", type=str, default="results/phase",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Parse parameter lists
    alphas = [float(x) for x in args.alphas.split(",")]
    etas = [float(x) for x in args.etas.split(",")]
    
    run_phase_sweep(alphas, etas, args.n, args.T, args.M, args.seed, args.out_dir)
    print("\nPhase sweep complete!")


if __name__ == "__main__":
    main()
