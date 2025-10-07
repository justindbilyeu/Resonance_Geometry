# tests/test_as_float_scalar.py
import math
import numpy as np
import pathlib, sys

# Make repo root importable when running locally (CI sets PYTHONPATH)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rg.validation.phase_boundary_fit import as_float_scalar


def test_scalar_int_float():
    assert as_float_scalar(3) == 3.0
    assert as_float_scalar(2.5) == 2.5

def test_numpy_scalar_types():
    assert as_float_scalar(np.float64(2.5)) == 2.5
    assert as_float_scalar(np.array(7.2)) == 7.2  # 0-d array

def test_last_finite_from_mixed():
    x = [np.nan, 1.1, np.inf, 2.2]
    assert as_float_scalar(x) == 2.2

def test_stringy_inputs():
    x = ["1.0", "2.0"]
    assert as_float_scalar(x) == 2.0

def test_nanmean_fallback():
    x = [1.0, np.nan, 3.0]
    assert as_float_scalar(x, prefer="mean") == 2.0

def test_no_finite_values():
    x = [np.nan, np.inf, -np.inf]
    assert as_float_scalar(x) == 0.0