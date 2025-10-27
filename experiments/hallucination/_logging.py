# experiments/hallucination/_logging.py
import csv
import os


def log_step_csv(path, t, eta, eta_eff, kappa, gain_term, I_bar, norm, lam_max, regime):
    """
    Log a single timestep to CSV with adaptive eta metrics.

    Args:
        path: CSV file path
        t: Time
        eta: Base eta parameter
        eta_eff: Effective eta (after adaptive gain)
        kappa: Condition number
        gain_term: Adaptive gain term
        I_bar: Moving average MI
        norm: Total norm (||omega_x|| + ||omega_y||)
        lam_max: Lambda_max stability estimate
        regime: Regime classification (0=grounded, 1=creative, 2=hallucinatory)
    """
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["t", "eta", "eta_eff", "kappa", "gain", "I_bar", "norm", "lambda_max", "regime"])
        w.writerow([t, eta, eta_eff, kappa, gain_term, I_bar, norm, lam_max, regime])
