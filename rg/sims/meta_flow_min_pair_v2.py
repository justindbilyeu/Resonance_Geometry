#!/usr/bin/env python3
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
