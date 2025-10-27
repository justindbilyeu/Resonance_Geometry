import numpy as np

def compute_effective_eta(eta: float, covariance_matrix: np.ndarray, epsilon: float = 1e-12,
                          cap: float = 15.0, dim_override: int | None = None):
    """
    Adaptive coupling strength with whitening gain:
      whitening_gain = log(kappa(Sigma))/d,  kappa = λ_max / λ_min
    eta_eff = eta * (1 + whitening_gain), tanh-capped and stabilized.

    Returns: (eta_eff_smoothed, kappa, whitening_gain_raw)
    """
    d = covariance_matrix.shape[0] if dim_override is None else dim_override

    Sigma_reg = covariance_matrix + epsilon * np.eye(covariance_matrix.shape[0])
    evals = np.linalg.eigvalsh(Sigma_reg)
    lam_max = float(np.max(evals))
    lam_min = float(max(np.min(evals), epsilon))
    kappa = lam_max / lam_min

    log_kappa = float(np.log(np.clip(kappa, 1.0, 1e12)))
    whitening_gain = log_kappa / max(d, 1)

    # cap via tanh to avoid runaway amplification for extreme kappa
    # maps [0, inf) → [0, 1), then scale to 'cap'
    eta_eff = eta * (1.0 + np.tanh(whitening_gain) * (cap / 15.0))
    return eta_eff, kappa, whitening_gain


class EtaEffEMA:
    """EMA smoother for eta_eff."""
    def __init__(self, alpha: float = 0.1, init: float | None = None):
        self.alpha = alpha
        self.value = init

    def update(self, x: float) -> float:
        self.value = (self.alpha * x) + ((1 - self.alpha) * (self.value if self.value is not None else x))
        return self.value
