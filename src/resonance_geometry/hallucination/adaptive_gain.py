# src/resonance_geometry/hallucination/adaptive_gain.py
import numpy as np

def compute_effective_eta(eta: float, Sigma: np.ndarray, epsilon: float = 1e-12,
                          tanh_cap: bool = True, d_scale: int | None = None) -> tuple[float, float, float]:
    """
    Adaptive eta with whitening gain from covariance conditioning.

    eta_eff = eta * (1 + log(kappa(Sigma))/d)      with kappa = lambda_max / lambda_min
    Optional cap: gain_term <- tanh( log_kappa / d_scale ), default d_scale = d
    Returns: (eta_eff, kappa, gain_term)

    Stable for near-singular Sigma via eigvalsh + eps clamp + log clip.
    """
    d = Sigma.shape[0]
    if d_scale is None or d_scale <= 0:
        d_scale = d

    # regularize (Hermitian if complex)
    if np.iscomplexobj(Sigma):
        Sigma = Sigma.real
    Sigma_reg = Sigma + epsilon * np.eye(d)

    evals = np.linalg.eigvalsh(Sigma_reg)
    lam_max = float(np.max(evals))
    lam_min = float(max(np.min(evals), epsilon))
    kappa = lam_max / lam_min

    # safe log kappa (>= 0)
    log_kappa = float(np.log(np.clip(kappa, 1.0, 1e12)))
    base_term = log_kappa / max(d, 1)
    gain_term = np.tanh(base_term / max(d_scale, 1)) if tanh_cap else base_term

    eta_eff = eta * (1.0 + gain_term)
    return eta_eff, kappa, gain_term


class EtaEffEMA:
    """EMA smoother for eta_eff to avoid step jitter."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._state = None

    def update(self, value: float) -> float:
        if self._state is None:
            self._state = value
        else:
            self._state = self.alpha * value + (1.0 - self.alpha) * self._state
        return self._state
