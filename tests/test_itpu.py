import math
from pathlib import Path
import importlib.util

# Load the compute_itpu.py module dynamically (no package import assumptions)
mod_path = Path("Phase_4_Falsification/analysis/compute_itpu.py").resolve()
spec = importlib.util.spec_from_file_location("compute_itpu", mod_path)
compute_itpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_itpu)  # type: ignore

def test_itpu_basic_identity():
    # ITPU = λ * Φ * (1 - κ)
    phi = 0.80
    kappa = 0.65
    lam = 0.78
    expected = lam * phi * (1 - kappa)
    assert abs(compute_itpu.itpu(phi, kappa, lam) - expected) < 1e-12

def test_itpu_monotonicity():
    # Increasing λ or Φ should not decrease ITPU; increasing κ should not increase ITPU
    base = compute_itpu.itpu(0.6, 0.4, 0.5)
    inc_lambda = compute_itpu.itpu(0.6, 0.4, 0.6)
    inc_phi = compute_itpu.itpu(0.7, 0.4, 0.5)
    inc_kappa = compute_itpu.itpu(0.6, 0.5, 0.5)

    assert inc_lambda >= base
    assert inc_phi >= base
    assert inc_kappa <= base
