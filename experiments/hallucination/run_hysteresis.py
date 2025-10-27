#!/usr/bin/env python3
"""
Hysteresis sweep for hallucination phase transitions.

Reads v2 config, performs forward/backward sweeps, outputs:
- docs/papers/neurips/figures/Geometric Theory of AI Hallucination/hysteresis_v2.png
- experiments/hallucination/results/hysteresis_metrics.json
"""
import os, argparse, json, numpy as np, matplotlib.pyplot as plt
import sys
from pathlib import Path
import yaml

# Add src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from resonance_geometry.hallucination.phase_dynamics import (
    simulate_trajectory,
    run_phase,
)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def sweep_hysteresis(eta_vals, params, small_scale=1.0, large_scale=3.0, T=60.0, dt=0.01):
    # Base operating vectors (same as simulator)
    base_x = np.array([0.12,0.08,0.05]); base_y = np.array([0.07,-0.11,0.04])

    # --- Up-sweep (continuation from small amplitude)
    up_norm, up_states = [], []
    ox, oy = small_scale*base_x, small_scale*base_y
    mi_bar = 0.0
    for eta in eta_vals:
        p = params.copy(); p['eta']=eta
        traj = simulate_trajectory(p, T=T, dt=dt, init_x=ox, init_y=oy, mi_bar0=mi_bar)
        ox, oy, mi_bar = traj['final_x'], traj['final_y'], traj['final_mi_bar']
        up_norm.append(traj['norm'][-1]); up_states.append((ox.copy(),oy.copy(),mi_bar))

    # --- Down-sweep (continuation from large amplitude onto high branch)
    dn_norm, dn_states = [], []
    ox, oy = large_scale*base_x, large_scale*base_y
    mi_bar = 1.0
    for eta in eta_vals[::-1]:
        p = params.copy(); p['eta']=eta
        traj = simulate_trajectory(p, T=T, dt=dt, init_x=ox, init_y=oy, mi_bar0=mi_bar)
        ox, oy, mi_bar = traj['final_x'], traj['final_y'], traj['final_mi_bar']
        dn_norm.append(traj['norm'][-1]); dn_states.append((ox.copy(),oy.copy(),mi_bar))
    dn_norm = dn_norm[::-1]

    gaps = np.abs(np.array(up_norm) - np.array(dn_norm))
    return np.array(up_norm), np.array(dn_norm), float(np.nanmax(gaps))

def find_jump_points(etas, up_norm, dn_norm, threshold=1.0):
    """Find indices where jumps occur (large derivative)."""
    up_deriv = np.diff(up_norm)
    dn_deriv = np.diff(dn_norm)

    up_jumps = np.where(np.abs(up_deriv) > threshold)[0]
    dn_jumps = np.where(np.abs(dn_deriv) > threshold)[0]

    return {
        'up_jump_etas': etas[up_jumps].tolist() if len(up_jumps) else [],
        'dn_jump_etas': etas[dn_jumps].tolist() if len(dn_jumps) else [],
    }


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description='Run hysteresis sweep')
    ap.add_argument('--config', type=str, required=True, help='Path to config YAML')
    ap.add_argument('--eta_min', type=float, default=0.2)
    ap.add_argument('--eta_max', type=float, default=5.0)
    ap.add_argument('--eta_steps', type=int, default=41)
    ap.add_argument('--lam', type=float, default=1.0, help='Fixed lambda for hysteresis sweep')
    args = ap.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Adaptive gain: {config.get('use_adaptive_gain', False)}")

    etas = np.linspace(args.eta_min, args.eta_max, args.eta_steps)

    params_base = {
        'lambda': args.lam,
        'gamma': config['gamma'],
        'k': 1.0,
        'alpha': config['alpha'],
        'beta': config['beta'],
        'skew': config['kappa'],
        'mu': 0.0,
        'mi_window': config['window'],
        'mi_ema': config['ema_alpha'],
        'omega_anchor': np.array([0.0, 0.0, 0.0]),
        'use_adaptive_gain': config.get('use_adaptive_gain', False),
    }

    # Hysteresis sweep (fixed λ,γ)
    print("\nRunning hysteresis sweep...")
    up, dn, gap = sweep_hysteresis(etas, params_base, small_scale=1.0, large_scale=3.0,
                                    T=config['t_horizon'] * 16, dt=config['dt'])
    print(f"Hysteresis max gap: {gap:.4f}")

    # Find jump points
    jump_info = find_jump_points(etas, up, dn, threshold=1.0)

    # Save metrics JSON
    metrics = {
        'max_gap': float(gap),
        'lambda': args.lam,
        'gamma': config['gamma'],
        'adaptive_gain': config.get('use_adaptive_gain', False),
        **jump_info,
    }

    metrics_path = Path('experiments/hallucination/results/hysteresis_metrics.json')
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot
    plt.figure(figsize=(9, 6))
    plt.plot(etas, up, 'b-', label='Up-sweep (small init)', linewidth=2)
    plt.plot(etas, dn, 'r-', label='Down-sweep (large init)', linewidth=2)
    plt.xlabel(r'$\eta$ (Resonance Gain)', fontsize=12)
    plt.ylabel(r'$\|\omega_x\| + \|\omega_y\|$ (Total Amplitude)', fontsize=12)
    title = f'Hysteresis @ λ={args.lam}, γ={config["gamma"]} (gap={gap:.3f})'
    if config.get('use_adaptive_gain', False):
        title += ' [v2: Adaptive Gain]'
    plt.title(title, fontsize=13)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=11)

    hyst_png = Path('docs/papers/neurips/figures/Geometric Theory of AI Hallucination/hysteresis_v2.png')
    hyst_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(hyst_png, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {hyst_png}")
    plt.close()

    print("\nHysteresis sweep complete!")

if __name__ == "__main__":
    main()
