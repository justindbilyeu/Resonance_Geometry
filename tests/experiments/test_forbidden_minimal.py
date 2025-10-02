import os

import numpy as np
from experiments.forbidden_region_detector import (
    compute_largest_forbidden_component,
    minimal_forbidden_test,
)


def test_minimal_forbidden_runs(tmp_path):
    out = tmp_path / "res"
    summ = minimal_forbidden_test(grid_res=4, n_random_runs=50, n=4, steps=50, out_dir=str(out))
    assert "forbidden_pct" in summ
    assert "largest_forbidden_component" in summ
    assert os.path.exists(out / "forbidden_summary.json")


def test_largest_component_uses_networkx():
    visited = np.ones((2, 2, 2, 2), dtype=bool)
    visited[(0, 0, 0, 0)] = False
    visited[(0, 0, 0, 1)] = False
    visited[(1, 1, 1, 1)] = False

    largest = compute_largest_forbidden_component(visited)
    assert largest == 2
