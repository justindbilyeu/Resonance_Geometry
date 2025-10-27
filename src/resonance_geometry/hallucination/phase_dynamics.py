#!/usr/bin/env python3
import os

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

def adaptive_gain_eta(eta_base: float, cov_full: np.ndarray, use_adaptive: bool, eps: float = 1e-9) -> float:
    """
    Adaptive MI gain (v2): η_eff = η * (1 + log(cond(Σ))/d)

    Args:
        eta_base: Base resonance gain parameter
        cov_full: Full covariance matrix (d x d)
        use_adaptive: If False, returns eta_base unchanged
        eps: Small value to prevent division by zero

    Returns:
        Effective eta value, boosted by covariance conditioning
    """
    if not use_adaptive:
        return eta_base
    eigvals = np.linalg.eigvalsh(cov_full)
    eigvals = np.clip(eigvals, eps, None)
    cond = float(eigvals.max() / eigvals.min())
    d = cov_full.shape[0]
    return eta_base * (1.0 + np.log(cond) / d)

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

    # Adaptive gain (v2 feature, gated by use_adaptive_gain flag)
    use_adaptive = params.get('use_adaptive_gain', False)
    eta_base = params['eta']
    if use_adaptive and len(hist) >= params.get('mi_window', 30):
        window = params.get('mi_window', 30)
        recent = np.array(hist[-window:])
        if recent.shape[0] >= 3:
            try:
                cov_full = np.cov(recent.T, bias=True)
                eta_eff = adaptive_gain_eta(eta_base, cov_full, use_adaptive)
            except Exception:
                eta_eff = eta_base
        else:
            eta_eff = eta_base
    else:
        eta_eff = eta_base

    # Create params with effective eta for RHS
    params_eff = params.copy()
    params_eff['eta'] = eta_eff

    k1x, k1y = rhs_pair(ox, oy, params_eff, mi_bar)
    px, py = ox + dt*k1x, oy + dt*k1y
    # second eval with provisional mi (update history with predicted state)
    hist_pred = hist + [np.concatenate([px,py])]
    mi_inst2 = compute_mi(hist_pred, window=params.get('mi_window',30))
    mi_bar2 = (1-ema)*mi_bar + ema*mi_inst2

    # Recompute adaptive gain for second step
    if use_adaptive and len(hist_pred) >= params.get('mi_window', 30):
        window = params.get('mi_window', 30)
        recent2 = np.array(hist_pred[-window:])
        if recent2.shape[0] >= 3:
            try:
                cov_full2 = np.cov(recent2.T, bias=True)
                eta_eff2 = adaptive_gain_eta(eta_base, cov_full2, use_adaptive)
            except Exception:
                eta_eff2 = eta_base
        else:
            eta_eff2 = eta_base
    else:
        eta_eff2 = eta_base

    params_eff2 = params.copy()
    params_eff2['eta'] = eta_eff2
    k2x, k2y = rhs_pair(px, py, params_eff2, mi_bar2)

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

def _resolve_seed(seed):
    env_seed = os.environ.get("RG_SEED")
    if env_seed is not None:
        try:
            return int(env_seed)
        except ValueError:
            pass
    if seed is None:
        return 1337
    return seed


def simulate_trajectory(params, T=60.0, dt=0.01, seed=None, init_x=None, init_y=None, mi_bar0=None):
    rng = np.random.default_rng(_resolve_seed(seed))
    # nonzero operating point (tiny bias)
    base_x = np.array([0.12, 0.08, 0.05], dtype=float)
    base_y = np.array([0.07,-0.11, 0.04], dtype=float)
    ox = base_x.copy() if init_x is None else init_x.copy()
    oy = base_y.copy() if init_y is None else init_y.copy()

    steps = int(T/dt) if dt > 0 else 0
    if os.environ.get("RG_CI"):
        max_steps = int(os.environ.get("RG_CI_MAX_STEPS", "200"))
        steps = min(steps, max_steps)
    steps = max(1, steps)
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
