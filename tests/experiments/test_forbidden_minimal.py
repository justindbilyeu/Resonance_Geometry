import os

from experiments.forbidden_region_detector import minimal_forbidden_test


def test_minimal_forbidden_runs(tmp_path):
    out = tmp_path / "res"
    summ = minimal_forbidden_test(grid_res=4, n_random_runs=50, n=4, steps=50, out_dir=str(out))
    assert "forbidden_pct" in summ
    assert os.path.exists(out / "forbidden_summary.json")
