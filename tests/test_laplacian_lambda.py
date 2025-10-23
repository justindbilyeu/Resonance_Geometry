import numpy as np
import pytest

pytest.importorskip("sklearn", reason="Requires scikit-learn for kNN graph")

from rg_empirical.laplacian_lambda import lambda_max_Lsym


def test_lambda_max_bounds_and_finiteness():
    # synthetic token embeddings: 40 tokens in 8D
    X = np.random.RandomState(0).randn(40, 8)
    lam = lambda_max_Lsym(X, k=10)
    assert np.isfinite(lam)
    assert 0.0 <= lam <= 2.0  # spectrum of L_sym is in [0,2]


def test_small_T_yields_nan():
    X = np.random.RandomState(1).randn(2, 8)  # too few tokens
    lam = lambda_max_Lsym(X, k=3)
    assert np.isnan(lam)
