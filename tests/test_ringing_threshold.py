# tests/test_ringing_threshold.py
import numpy as np
from simulations.ringing_threshold import GPParams, kc_engineering, solve_omega_c

def test_kc_engineering_monotone_in_delay():
    p = GPParams(A=0.5, B=1.0, Delta=0.05)
    kc1 = kc_engineering(p)
    p.Delta = 0.2
    kc2 = kc_engineering(p)
    assert kc2 > kc1  # more delay → lower phase margin → higher Kc

def test_deepseek_examples_rough_match():
    # DeepSeek: (A,B,Delta)->Kc ≈ {14.21, 5.20, 8.31} with ~±20% error
    ex = [
        (0.1, 1.0, 0.1, 14.21),
        (1.0, 1.0, 0.1, 5.20),
        (0.1, 1.0, 0.5, 8.31),
    ]
    for A,B,Delta,Kc_ref in ex:
        kc = kc_engineering(GPParams(A=A,B=B,Delta=Delta))
        rel_err = abs(kc - Kc_ref)/Kc_ref
        assert rel_err < 0.25  # allow 25% slack
