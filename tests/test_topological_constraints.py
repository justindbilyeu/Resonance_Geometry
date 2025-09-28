"""Placeholder smoke tests for the topological constraint pipeline stubs."""

import numpy as np

def test_state_vector_extraction_shape():
    from resonance_geometry.state_vector import extract_state_vector

    g = np.eye(3)
    L = np.eye(3)
    mi = np.array([0.1, 0.2, 0.3])
    vec = extract_state_vector(1.0, 0.5, 0.2, g, mi, L=L)
    assert vec.shape == (6,)
    assert np.isfinite(vec).all()


def test_forbidden_classifier_label():
    from resonance_geometry.forbidden import classify_cell_accessibility

    label = classify_cell_accessibility([0] * 6, n_random_starts=1, n_adversarial=1)
    assert label in {"REACHABLE", "RARE_BUT_REACHABLE", "FORBIDDEN"}


def test_fractal_dimension_placeholder():
    from resonance_geometry.fractals import measure_fractal_dimension

    points = np.random.rand(150, 6)
    dim, ci = measure_fractal_dimension(points)
    assert isinstance(dim, float)
    assert isinstance(ci, tuple) and len(ci) == 2
