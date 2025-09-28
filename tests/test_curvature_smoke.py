import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_curvature_smoke(tmp_path):
    g = np.array(
        [
            [0.0, 0.8, 0.2, 0.0],
            [0.8, 0.0, 0.5, 0.3],
            [0.2, 0.5, 0.0, 0.7],
            [0.0, 0.3, 0.7, 0.0],
        ],
        dtype=float,
    )
    coupling_path = tmp_path / "sample.npy"
    np.save(coupling_path, g)
    mask = np.array([True, True, True, False])
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)

    output_dir = tmp_path / "outputs"
    figure_dir = tmp_path / "figures"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "experiments/analysis/curvature_signatures.py",
        "--input",
        str(coupling_path),
        "--boundary-mask",
        str(mask_path),
        "--output-dir",
        str(output_dir),
        "--figure-dir",
        str(figure_dir),
    ]
    subprocess.check_call(cmd, cwd=repo_root)

    summary_path = output_dir / "curvature_report.json"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    for key in [
        "boundary_cell_count",
        "kappa_pass_fraction",
        "gradient_pass_fraction",
        "pass",
    ]:
        assert key in report
    assert (figure_dir / "curvature_hist.png").exists()
