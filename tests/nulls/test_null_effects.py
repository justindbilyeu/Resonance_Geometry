import importlib

import numpy as np
from numpy.fft import fft
import pytest

networkx = pytest.importorskip("networkx")

null_models = importlib.import_module("experiments.topo_test.05_null_models")
phase_shuffle = null_models.phase_shuffle
rewire_graph = null_models.rewire_graph


def test_null_models_preserve_invariants():
    series = np.sin(np.linspace(0, 10, 100))
    null = phase_shuffle(series)
    assert np.allclose(np.abs(fft(series)), np.abs(fft(null)), atol=1e-5)
    assert not np.allclose(series, null)
    graph = networkx.grid_graph((3, 3))
    null_graph = rewire_graph(graph)
    assert sorted(dict(graph.degree()).values()) == sorted(dict(null_graph.degree()).values())
    assert len(graph.edges) == len(null_graph.edges)
