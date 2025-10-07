#!/usr/bin/env python3
<<<<<<< HEAD
"""Lightweight placeholder simulation harness with algebra/backends."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class Trajectory:
    """Container mirroring the dictionary interface used downstream."""

    t: np.ndarray
    lambda_max: np.ndarray
    regime: int
    I_bar: float
    omega_norm_final: float
    params: Dict[str, float]
    algebra: str
    antisym: bool
    noise_std: float
    mi_est: str
    mi_scale: float
    seed: int

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def _classify_regime(lam: float, eta: float, gamma: float) -> int:
    """Toy classifier for grounded/creative/hallucinatory regimes."""
    boundary = lam + gamma
    if eta > boundary:
        return 2  # hallucinatory
    if eta > lam:
        return 1  # marginal/creative
    return 0  # grounded


def _compute_force(omega_x: np.ndarray, omega_y: np.ndarray, algebra: str) -> np.ndarray:
    """Return interaction term for the chosen algebra backend."""
    if algebra == 'so3':
        # TODO: consider re-introducing a scale factor if future fits require it.
        return np.cross(omega_x, omega_y)
    # Default "su2" pipeline mimics the commutator behaviour with a 1/2 factor.
    return 0.5 * np.cross(omega_x, omega_y)


def _derivatives(
    omega_x: np.ndarray,
    omega_y: np.ndarray,
    params: Dict[str, float],
    algebra: str,
    antisym: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    lam = float(params.get('lambda', 1.0))
    eta = float(params.get('eta', 1.0))
    gamma = float(params.get('gamma', 0.5))
    alpha = float(params.get('alpha', 0.6))
    beta = float(params.get('beta', 0.02))
    skew = float(params.get('skew', 0.12))
    k = float(params.get('k', 1.0))
    anchor = np.asarray(params.get('omega_anchor', np.zeros(3)))

    force = _compute_force(omega_x, omega_y, algebra)
    coupling = k * force
    if antisym:
        coupling_x = coupling
        coupling_y = -coupling
    else:
        coupling_x = coupling
        coupling_y = coupling

    # Mildly damped coupling dynamics against an anchor.
    drift_x = alpha * (anchor - omega_x) - lam * omega_x + eta * omega_y - beta * np.cross(omega_x, anchor)
    drift_y = alpha * (anchor - omega_y) - gamma * omega_y + lam * omega_x - skew * np.cross(anchor + omega_y, omega_x)

    return drift_x + coupling_x, drift_y + coupling_y


def simulate_trajectory(params: Dict[str, float], T_max: float = 3.0, dt: float = 0.01) -> Trajectory:
    """Return a lightweight trajectory with qualitative behaviour."""

    lam = float(params.get('lambda', 1.0))
    eta = float(params.get('eta', 1.0))
    gamma = float(params.get('gamma', 0.5))

    algebra = str(params.get('algebra', 'su2')).lower()
    if algebra not in {'su2', 'so3'}:
        raise ValueError(f"Unsupported algebra '{algebra}'")
    antisym = bool(params.get('antisym_coupling', False))
    noise_std = float(params.get('noise_std', 0.0))
    mi_est = str(params.get('mi_est', 'corr')).lower()
    if mi_est not in {'corr', 'svd'}:
        raise ValueError(f"Unsupported MI estimator '{mi_est}'")
    mi_scale = float(params.get('mi_scale', 1.0))
    seed = int(params.get('seed', 42))
    mi_window = int(params.get('mi_window', 30))

    t = np.arange(0.0, T_max + 0.5 * dt, dt)
    steps = len(t)

    anchor = np.asarray(params.get('omega_anchor', np.zeros(3)))
    omega_x = anchor + np.array([lam, eta, gamma]) * 0.1
    omega_y = anchor + np.array([eta, gamma, lam]) * 0.08

    rng = np.random.default_rng(seed) if noise_std > 0.0 else None

    omega_x_hist = np.zeros((steps, 3))
    omega_y_hist = np.zeros((steps, 3))
    lambda_max = np.zeros(steps)

    for idx, time in enumerate(t):
        omega_x_hist[idx] = omega_x
        omega_y_hist[idx] = omega_y

        interaction = np.dot(omega_x, omega_y)
        envelope = np.tanh((idx + 1) * dt * float(params.get('k', 1.0)))
        offset = eta - lam - gamma / 2.0
        lambda_max[idx] = envelope * offset + 0.05 * interaction

        dx1, dy1 = _derivatives(omega_x, omega_y, params, algebra, antisym)
        trial_x = omega_x + dt * dx1
        trial_y = omega_y + dt * dy1
        dx2, dy2 = _derivatives(trial_x, trial_y, params, algebra, antisym)

        omega_x = omega_x + 0.5 * dt * (dx1 + dx2)
        omega_y = omega_y + 0.5 * dt * (dy1 + dy2)

        if noise_std > 0.0 and rng is not None:
            omega_x = omega_x + noise_std * rng.normal(size=omega_x.shape)
            omega_y = omega_y + noise_std * rng.normal(size=omega_y.shape)

    # Mutual information style summary statistics.
    history_len = min(mi_window, steps)
    recent_x = omega_x_hist[-history_len:]
    recent_y = omega_y_hist[-history_len:]

    if mi_est == 'corr':
        stacked = np.vstack([recent_x.ravel(), recent_y.ravel()])
        if stacked.shape[1] < 2:
            I_bar = 0.0
        else:
            corr = np.corrcoef(stacked)[0, 1]
            I_bar = float(np.log1p(abs(corr)))
    else:  # svd estimator
        recent = np.concatenate([recent_x, recent_y], axis=1)
        if recent.size == 0:
            I_bar = 0.0
        else:
            singular_values = np.linalg.svd(recent, compute_uv=False)
            r = int(min(3, singular_values.shape[0]))
            if r == 0:
                I_bar = 0.0
            else:
                I_bar = float(np.log(1.0 + np.sum(singular_values[:r])) / r)

    I_bar *= mi_scale

    regime = _classify_regime(lam, eta, gamma)
    omega_norm_final = float(0.5 * (np.linalg.norm(omega_x) + np.linalg.norm(omega_y)))

    trajectory = Trajectory(
        t=t,
        lambda_max=lambda_max,
        regime=regime,
        I_bar=I_bar,
        omega_norm_final=omega_norm_final,
        params=dict(params),
        algebra=algebra,
        antisym=antisym,
        noise_std=noise_std,
        mi_est=mi_est,
        mi_scale=mi_scale,
        seed=seed,
    )
    return trajectory


def batch_simulate(grid: Iterable[Dict[str, float]], **kwargs) -> Iterable[Trajectory]:
    """Helper to simulate a collection of parameter dictionaries."""
    for params in grid:
        merged = dict(params)
        merged.update(kwargs)
        yield simulate_trajectory(merged)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic meta-flow pair simulator")
    parser.add_argument('--lambda', dest='lam', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--T_max', type=float, default=3.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--algebra', choices=['su2', 'so3'], default='su2')
    parser.add_argument('--antisym_coupling', action='store_true', default=False)
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mi_est', choices=['corr', 'svd'], default='corr')
    parser.add_argument('--mi_scale', type=float, default=1.0)
    parser.add_argument('--mi_window', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.02)
    parser.add_argument('--skew', type=float, default=0.12)
    parser.add_argument('--k', type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    params = {
        'lambda': args.lam,
        'eta': args.eta,
        'gamma': args.gamma,
        'algebra': args.algebra,
        'antisym_coupling': args.antisym_coupling,
        'noise_std': args.noise_std,
        'seed': args.seed,
        'mi_est': args.mi_est,
        'mi_scale': args.mi_scale,
        'mi_window': args.mi_window,
        'alpha': args.alpha,
        'beta': args.beta,
        'skew': args.skew,
        'k': args.k,
    }
    traj = simulate_trajectory(params, T_max=args.T_max, dt=args.dt)
    print(f"algebra={traj.algebra}, antisym={traj.antisym}, noise={traj.noise_std:.3f}")
    print(f"λ_max(final)={traj.lambda_max[-1]:.4f}, Ī={traj.I_bar:.4f}, ‖ω‖≈{traj.omega_norm_final:.4f}")


if __name__ == '__main__':
    main()
=======
import numpy as np, matplotlib.pyplot as plt

# ----- SU(2) helpers -----
sigma_x = np.array([[0, 1],[1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0],[0, -1]], dtype=complex)
PAULI = [sigma_x, sigma_y, sigma_z]

def omega_to_matrix(v):
    return 0.5j * (v[0]*PAULI[0] + v[1]*PAULI[1] + v[2]*PAULI[2])

def matrix_to_vec(M):
    return np.array([
        -2*np.imag(M[0,1]),
        2*np.real(M[0,1]),
        -2*np.imag(M[0,0])
    ])

def commutator(A,B): return A@B - B@A
def curvature_F(ox, oy): return commutator(omega_to_matrix(ox), omega_to_matrix(oy))
def F_to_vec(F): return matrix_to_vec(F)

# --- MI estimation (instantaneous, from temporal window over 6 comps) ---
def compute_mi(history, window=30):
    if len(history) < window: return 0.0
    recent = np.array(history[-window:])   # (window, 6)
    if recent.shape[0] < 3: return 0.0
    try:
        corr = np.corrcoef(recent.T)       # (6,6)
        rho2 = np.clip(corr**2, 0, 0.999)
        return float(-0.5 * np.mean(np.log(1 - rho2 + 1e-12)))
    except Exception:
        return 0.0

# --- RHS with linear MI gain, cubic–quintic, and skew ---
def rhs_pair(ox, oy, params, mi_bar):
    eta, lam, gamma, k = params['eta'], params['lambda'], params['gamma'], params['k']
    alpha = params.get('alpha', 0.6)
    beta  = params.get('beta',  0.02)
    skew  = params.get('skew',  0.12)
    anchor = params.get('omega_anchor', np.zeros(3))
    mu = params.get('mu', 0.0)  # legacy cubic saturation (off by default)

    # Curvature piece (shared)
    F = curvature_F(ox, oy)
    F_vec = F_to_vec(F)
    curv_push_x = -0.5 * F_vec
    curv_push_y = -0.5 * F_vec

    # Linear MI gain (with EMA mi_bar)
    drive_x = eta * mi_bar * ox
    drive_y = eta * mi_bar * oy

    # Grounding + damping
    gradUx = k * (ox - anchor); gradUy = k * (oy - anchor)
    damp_x = gamma * ox;         damp_y = gamma * oy

    # Cubic–quintic amplitude law (subcritical window for hysteresis)
    r2x, r2y = np.dot(ox,ox), np.dot(oy,oy)
    r4x, r4y = r2x*r2x, r2y*r2y
    nl_x = + alpha * r2x * ox - beta * r4x * ox
    nl_y = + alpha * r2y * oy - beta * r4y * oy

    # Optional legacy cubic saturation
    if mu != 0.0:
        nl_x -= mu * r2x * ox
        nl_y -= mu * r2y * oy

    # Small non-normal skew coupling
    skew_x =  skew * oy
    skew_y = -skew * ox

    dox = curv_push_x + drive_x - lam*gradUx - damp_x + nl_x + skew_x
    doy = curv_push_y + drive_y - lam*gradUy - damp_y + nl_y + skew_y
    return dox, doy

def heun_step_pair(ox, oy, params, mi_bar, hist, dt):
    # instant MI from history; EMA for memory
    mi_inst = compute_mi(hist, window=params.get('mi_window', 30))
    ema = params.get('mi_ema', 0.1)  # 0<ema<=1
    mi_bar = (1-ema)*mi_bar + ema*mi_inst

    k1x, k1y = rhs_pair(ox, oy, params, mi_bar)
    px, py = ox + dt*k1x, oy + dt*k1y
    # second eval with provisional mi (update history with predicted state)
    hist_pred = hist + [np.concatenate([px,py])]
    mi_inst2 = compute_mi(hist_pred, window=params.get('mi_window',30))
    mi_bar2 = (1-ema)*mi_bar + ema*mi_inst2
    k2x, k2y = rhs_pair(px, py, params, mi_bar2)

    new_x = ox + 0.5*dt*(k1x + k2x)
    new_y = oy + 0.5*dt*(k1y + k2y)
    new_mi_bar = mi_bar2
    return new_x, new_y, new_mi_bar

def estimate_lambda_max_simple(ox, oy, params, mi_bar):
    # crude: leading real part ≈ η*MĪ - λ*k - γ - 3μ‖ω‖²
    eta, lam, gamma, k = params['eta'], params['lambda'], params['gamma'], params['k']
    mu = params.get('mu', 0.0)
    norm_sq = np.dot(ox,ox) + np.dot(oy,oy)
    return eta*mi_bar - lam*k - gamma - 3*mu*norm_sq

def simulate_trajectory(params, T=60.0, dt=0.01, seed=0, init_x=None, init_y=None, mi_bar0=None):
    rng = np.random.default_rng(seed)
    # nonzero operating point (tiny bias)
    base_x = np.array([0.12, 0.08, 0.05], dtype=float)
    base_y = np.array([0.07,-0.11, 0.04], dtype=float)
    ox = base_x.copy() if init_x is None else init_x.copy()
    oy = base_y.copy() if init_y is None else init_y.copy()

    steps = int(T/dt)
    t_hist, E_hist, lam_hist, mi_hist, norm_hist = [], [], [], [], []
    hist = [np.concatenate([ox,oy])]
    mi_bar = compute_mi(hist, window=params.get('mi_window',30)) if mi_bar0 is None else mi_bar0

    for s in range(steps):
        ox, oy, mi_bar = heun_step_pair(ox, oy, params, mi_bar, hist, dt)
        hist.append(np.concatenate([ox,oy]))
        F = curvature_F(ox, oy)
        E_dual = np.linalg.norm(F, 'fro')**2
        lam_est = estimate_lambda_max_simple(ox, oy, params, mi_bar)
        nn = np.linalg.norm(ox) + np.linalg.norm(oy)

        t_hist.append(s*dt); E_hist.append(E_dual); lam_hist.append(lam_est)
        mi_hist.append(mi_bar); norm_hist.append(nn)

        if nn > 200 or E_dual > 1e6: break

    return {
        't': np.array(t_hist),
        'E_dual': np.array(E_hist),
        'lambda_max': np.array(lam_hist),
        'MI_bar': np.array(mi_hist),
        'norm': np.array(norm_hist),
        'final_x': ox, 'final_y': oy, 'final_mi_bar': mi_bar
    }

def classify_regime(traj):
    if not len(traj['norm']): return 0
    E, lam, nn = traj['E_dual'][-1], traj['lambda_max'][-1], traj['norm'][-1]
    if nn > 50 or E > 1.0 or lam > 0.1: return 2
    if -0.1 < lam < 0.1: return 1
    return 0

def run_phase(etas, lambdas, params_base, out_png='phase_diagram_v2.png'):
    phase = np.zeros((len(lambdas), len(etas)), dtype=int)
    for i, lam in enumerate(lambdas):
        for j, eta in enumerate(etas):
            p = params_base.copy(); p['eta']=eta; p['lambda']=lam
            traj = simulate_trajectory(p, T=40.0, dt=0.01)
            phase[i,j] = classify_regime(traj)
            print(f"η={eta:.2f} λ={lam:.2f} → regime={phase[i,j]} (λmax≈{traj['lambda_max'][-1]:.3f})")
    plt.figure(figsize=(8,6))
    extent=[etas[0], etas[-1], lambdas[0], lambdas[-1]]
    plt.imshow(phase, origin='lower', extent=extent, cmap='RdYlGn_r', vmin=0, vmax=2, aspect='auto')
    plt.colorbar(label='Regime (0=Grounded,1=Creative,2=Hallucinatory)')
    e = np.linspace(etas[0], etas[-1], 200)
    gamma = params_base['gamma']
    # Guideline assuming I≈1; if you estimate I_eff, replace accordingly
    plt.plot(e, e - (gamma), 'w--', lw=2, label=r'$\eta I \approx \lambda+\gamma$ (I≈1)')
    plt.xlabel(r'$\eta$'); plt.ylabel(r'$\lambda$'); plt.title('Phase Diagram — Meta-Resonance v2')
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150)
    return phase, out_png

if __name__ == "__main__":
    # simple smoke test
    params = {'lambda':1.0, 'gamma':0.5, 'k':1.0, 'alpha':0.6, 'beta':0.02, 'skew':0.12, 'mu':0.0, 'mi_window':30, 'mi_ema':0.1, 'omega_anchor':np.zeros(3), 'eta':2.0}
    traj = simulate_trajectory(params)
    print("Final norm:", traj['norm'][-1])
>>>>>>> origin/feature/hysteresis-and-boundary
