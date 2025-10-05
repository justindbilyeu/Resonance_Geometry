#!/usr/bin/env python3
import argparse, os, numpy as np, matplotlib.pyplot as plt
from rg.sims.meta_flow_min_pair_v2 import simulate_trajectory

def hallucinatory_eta_for_lambda(lam, eta_grid, base):
    """
    Sweep eta for fixed lambda, return first eta whose trajectory is hallucinatory.
    Hallucinatory if lambda_max > 0 (preferred) else large final norm as fallback.
    """
    for eta in eta_grid:
        params = dict(base); params['eta'] = float(eta); params['lambda'] = float(lam)
        traj = simulate_trajectory(params)  # rely on defaults (dt, steps) inside
        lam_series = traj.get('lambda_max', [])
        if isinstance(lam_series, (list, tuple)) and len(lam_series) > 0:
            lam_max = float(lam_series[-1])
        else:
            lam_max = float(traj.get('lambda_max', 0.0))

        # primary criterion: spectral instability
        if lam_max > 0.0:
            return float(eta)

        # fallback: if we can't get lambda_max reliably, use growth of norm
        norm_series = traj.get('norm', [])
        if isinstance(norm_series, (list, tuple)) and len(norm_series) > 0:
            if float(norm_series[-1]) > 20.0:  # big growth => unstable
                return float(eta)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--skew", type=float, default=0.12)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--mi_window", type=int, default=30)
    ap.add_argument("--mi_ema", type=float, default=0.1)
    ap.add_argument("--lam_min", type=float, default=0.1)
    ap.add_argument("--lam_max", type=float, default=5.0)
    ap.add_argument("--lam_steps", type=int, default=11)
    ap.add_argument("--eta_min", type=float, default=0.2)
    ap.add_argument("--eta_max", type=float, default=5.0)
    ap.add_argument("--eta_steps", type=int, default=101)
    args = ap.parse_args()

    outdir = "rg/results/sage_corrected"; os.makedirs(outdir, exist_ok=True)

    base = {
        'gamma': args.gamma, 'k': args.k, 'alpha': args.alpha, 'beta': args.beta,
        'skew': args.skew, 'mu': args.mu, 'mi_window': args.mi_window,
        'mi_ema': args.mi_ema, 'omega_anchor': np.zeros(3)
    }

    lam_grid = np.linspace(args.lam_min, args.lam_max, args.lam_steps)
    eta_grid = np.linspace(args.eta_min, args.eta_max, args.eta_steps)

    rows = []
    for lam in lam_grid:
        eta_c = hallucinatory_eta_for_lambda(lam, eta_grid, base)
        rows.append((lam, eta_c))
        print(f"λ={lam:.2f} → η_c={eta_c}")

    # save CSV
    csv_path = os.path.join(outdir, "phase_boundary.csv")
    with open(csv_path, "w") as f:
        f.write("lambda,eta_c\n")
        for lam, eta_c in rows:
            f.write(f"{lam},{'' if eta_c is None else eta_c}\n")
    print("Saved", csv_path)

    # fit η_c ≈ m·λ + b on available points
    pts = [(lam, e) for lam, e in rows if e is not None]
    if len(pts) >= 2:
        L = np.array([p[0] for p in pts])
        E = np.array([p[1] for p in pts])
        m, b = np.linalg.lstsq(np.vstack([L, np.ones_like(L)]).T, E, rcond=None)[0]
        pred = m*L + b
        ss_res = np.sum((E - pred)**2)
        ss_tot = np.sum((E - np.mean(E))**2) + 1e-12
        r2 = 1 - ss_res/ss_tot
        print(f"Fit η_c ≈ m·λ + b  →  m={m:.3f}, b={b:.3f}, R²={r2:.3f}")

        plt.figure(figsize=(6,5))
        plt.scatter(L, E, s=28, label="boundary points")
        Ld = np.linspace(L.min(), L.max(), 200)
        plt.plot(Ld, m*Ld + b, label=f"fit: η={m:.2f}λ+{b:.2f} (R²={r2:.2f})")
        plt.plot(Ld, Ld + args.gamma, 'k--', alpha=0.6, label=f"guide: η=λ+γ, γ={args.gamma}")
        plt.xlabel("λ (grounding)"); plt.ylabel("η_c (critical resonance)")
        plt.title("Phase Boundary Fit")
        plt.legend()
        figp = os.path.join(outdir, "phase_boundary_fit.png")
        plt.tight_layout(); plt.savefig(figp, dpi=150)
        print("Saved", figp)
    else:
        print("Insufficient points for a fit.")

if __name__ == "__main__":
    main()
