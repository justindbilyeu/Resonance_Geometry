import os

from experiments.forbidden_region_detector import minimal_forbidden_test
from experiments.adversarial_forcing import adversarial_attack_pipeline


def test_adversarial_pipeline(tmp_path):
    out = tmp_path / "res"
    summ = minimal_forbidden_test(grid_res=4, n_random_runs=50, n=4, steps=30, out_dir=str(out))
    # ensure visited file exists
    assert (out / "visited_4d.npy").exists()
    rep = adversarial_attack_pipeline(
        forbidden_summary_path=str(out / "forbidden_summary.json"),
        visited_path=str(out / "visited_4d.npy"),
        out_path=str(out / "adv.json"),
        max_forbidden_to_test=2,
        strategy_attempts=5,
    )
    assert "decision_hint" in rep
    assert os.path.exists(out / "adv.json")
