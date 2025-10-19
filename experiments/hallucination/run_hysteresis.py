#!/usr/bin/env python3
import os, argparse, numpy as np, matplotlib.pyplot as plt
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eta_min', type=float, default=0.2)
    ap.add_argument('--eta_max', type=float, default=5.0)
    ap.add_argument('--eta_steps', type=int, default=41)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--beta', type=float, default=0.02)
    ap.add_argument('--skew', type=float, default=0.12)
    ap.add_argument('--k', type=float, default=1.0)
    ap.add_argument('--mi_window', type=int, default=30)
    ap.add_argument('--mi_ema', type=float, default=0.1)
    ap.add_argument('--outdir', type=str, default='rg/results/sage_corrected')
    args = ap.parse_args()

    ensure_dir(args.outdir)
    etas = np.linspace(args.eta_min, args.eta_max, args.eta_steps)

    params_base = {
        'lambda': args.lam, 'gamma': args.gamma, 'k': args.k,
        'alpha': args.alpha, 'beta': args.beta, 'skew': args.skew,
        'mu': 0.0, 'mi_window': args.mi_window, 'mi_ema': args.mi_ema,
        'omega_anchor': np.array([0.0,0.0,0.0])
    }

    # Phase diagram
    lambdas = np.linspace(0.1, 5.0, 11)
    phase, phase_png = run_phase(etas, lambdas, params_base, out_png=os.path.join(args.outdir, 'phase_diagram_v2.png'))
    print(f"Saved {phase_png}")

    # Hysteresis sweep (fixed λ,γ)
    up, dn, gap = sweep_hysteresis(etas, params_base, small_scale=1.0, large_scale=3.0, T=80.0, dt=0.01)
    print(f"Hysteresis max gap: {gap:.4f}")

    plt.figure(figsize=(8,5))
    plt.plot(etas, up, label='Up-sweep (small init)')
    plt.plot(etas, dn, label='Down-sweep (large init)')
    plt.xlabel(r'$\eta$'); plt.ylabel(r'$\|\omega_x\|+\|\omega_y\|$')
    plt.title(f'Hysteresis @ λ={args.lam}, γ={args.gamma} (gap={gap:.3f})')
    plt.grid(alpha=.3); plt.legend(); 
    hyst_png = os.path.join(args.outdir, 'hysteresis_v2.png')
    plt.tight_layout(); plt.savefig(hyst_png, dpi=150)
    print(f"Saved {hyst_png}")

if __name__ == "__main__":
    main()
