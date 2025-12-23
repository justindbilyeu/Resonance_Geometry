"""
Test Procrustes shape matching (Fix #2).

Ensures deterministic shape matching for Procrustes alignment.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.f_ai.transport import (
    match_shapes_deterministic,
    procrustes_alignment,
    compute_edge_transport,
)


def test_match_shapes_equal_size():
    """Test shape matching when both have same size."""
    V_r = np.random.randn(50, 10)
    V_q = np.random.randn(50, 10)

    V_r_matched, V_q_matched = match_shapes_deterministic(
        V_r, V_q, m_max=100, method="first"
    )

    assert V_r_matched.shape == (50, 10)
    assert V_q_matched.shape == (50, 10)

    # Should be first 50 sentences
    np.testing.assert_array_equal(V_r_matched, V_r[:50])
    np.testing.assert_array_equal(V_q_matched, V_q[:50])


def test_match_shapes_unequal_size():
    """Test shape matching when sizes differ."""
    V_r = np.random.randn(100, 10)  # 100 sentences
    V_q = np.random.randn(30, 10)  # 30 sentences

    V_r_matched, V_q_matched = match_shapes_deterministic(
        V_r, V_q, m_max=100, method="first"
    )

    # Should take min(100, 30, 100) = 30
    assert V_r_matched.shape == (30, 10)
    assert V_q_matched.shape == (30, 10)

    # Check it's the first 30
    np.testing.assert_array_equal(V_r_matched, V_r[:30])
    np.testing.assert_array_equal(V_q_matched, V_q[:30])


def test_match_shapes_with_m_max():
    """Test shape matching with m_max constraint."""
    V_r = np.random.randn(200, 10)
    V_q = np.random.randn(150, 10)

    V_r_matched, V_q_matched = match_shapes_deterministic(
        V_r, V_q, m_max=50, method="first"
    )

    # Should take min(200, 150, 50) = 50
    assert V_r_matched.shape == (50, 10)
    assert V_q_matched.shape == (50, 10)


def test_match_shapes_seeded_method():
    """Test seeded random sampling is deterministic."""
    V_r = np.random.randn(100, 10)
    V_q = np.random.randn(100, 10)

    # Same seed should give same result
    V_r_1, V_q_1 = match_shapes_deterministic(
        V_r, V_q, m_max=50, method="seeded", seed=42
    )
    V_r_2, V_q_2 = match_shapes_deterministic(
        V_r, V_q, m_max=50, method="seeded", seed=42
    )

    np.testing.assert_array_equal(V_r_1, V_r_2)
    np.testing.assert_array_equal(V_q_1, V_q_2)

    # Different seed should give different result (probabilistically)
    V_r_3, V_q_3 = match_shapes_deterministic(
        V_r, V_q, m_max=50, method="seeded", seed=123
    )

    # Very unlikely to be exactly the same
    assert not np.array_equal(V_r_1, V_r_3) or not np.array_equal(V_q_1, V_q_3)


def test_procrustes_with_matched_shapes():
    """Test Procrustes works with matched shapes."""
    # Create two point clouds
    V_r = np.random.randn(50, 10)
    V_q = np.random.randn(50, 10)

    # Procrustes should work without error
    R, residual = procrustes_alignment(V_r, V_q)

    assert R.shape == (10, 10)
    assert residual >= 0.0

    # R should be orthogonal
    np.testing.assert_allclose(R @ R.T, np.eye(10), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_procrustes_raises_on_shape_mismatch():
    """Test Procrustes raises error on mismatched shapes."""
    V_r = np.random.randn(50, 10)
    V_q = np.random.randn(30, 10)  # Different number of points

    with pytest.raises(ValueError, match="Shape mismatch"):
        procrustes_alignment(V_r, V_q)


def test_edge_transport_handles_different_sizes():
    """Test compute_edge_transport handles different sentence counts (Fix #2)."""
    V_r = np.random.randn(100, 10)
    V_q = np.random.randn(30, 10)

    # Should not raise error
    result = compute_edge_transport(V_r, V_q, m_max=100)

    assert "R" in result
    assert "residual" in result
    assert "m" in result

    # Should have used 30 sentences (the minimum)
    assert result["m"] == 30


def test_procrustes_identity_for_identical_clouds():
    """Test Procrustes gives near-identity for identical point clouds."""
    V = np.random.randn(50, 10)

    R, residual = procrustes_alignment(V, V.copy())

    # Should be very close to identity
    np.testing.assert_allclose(R, np.eye(10), atol=1e-4)

    # Residual should be near zero
    assert residual < 1e-6


def test_procrustes_rotation_invariance():
    """Test Procrustes correctly recovers known rotation."""
    # Create random orthogonal matrix
    from scipy.stats import ortho_group

    d = 10
    R_true = ortho_group.rvs(d)

    # Create point cloud and rotated version
    V_r = np.random.randn(50, d)
    V_q = V_r @ R_true  # Rotate V_r

    # Procrustes should recover R_true
    R_estimated, residual = procrustes_alignment(V_r, V_q)

    # Should be very close to R_true (up to centering/normalization)
    # Residual should be very small
    assert residual < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
