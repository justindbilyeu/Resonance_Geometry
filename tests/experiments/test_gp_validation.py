"""Regression tests for GP validation utilities."""

import numpy as np

from scripts.gp_validation import apply_multiple_comparisons_correction


def test_apply_multiple_comparisons_correction_unsorted_input() -> None:
    """Benjamini–Hochberg correction should respect original ordering."""

    p_values = np.array([0.04, 0.001, 0.02])
    corrected = apply_multiple_comparisons_correction(p_values, method="fdr_bh")

    np.testing.assert_allclose(corrected, np.array([0.04, 0.003, 0.03]))
