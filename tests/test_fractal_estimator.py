from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.analysis.measure_fractal_boundary import main


def _sierpinski(level: int) -> np.ndarray:
    size = 3**level
    grid = np.ones((size, size), dtype=bool)

    def carve(x0: int, y0: int, span: int) -> None:
        if span == 1:
            return
        step = span // 3
        grid[x0 + step : x0 + 2 * step, y0 + step : y0 + 2 * step] = False
        for dx in range(3):
            for dy in range(3):
                if dx == 1 and dy == 1:
                    continue
                carve(x0 + dx * step, y0 + dy * step, step)

    carve(0, 0, size)
    return grid


def test_fractal_pipeline(tmp_path: Path) -> None:
    level = 4  # 3**4 = 81, gives plenty of boundary points
    base = tmp_path
    input_path = base / "outputs" / "topo" / "forbidden.json"
    output_path = base / "outputs" / "topo" / "fractal_dim.json"
    figure_path = base / "figures" / "topo" / "boxcount_fit.png"

    mask_2d = _sierpinski(level)
    shape = (mask_2d.shape[0], mask_2d.shape[1], 1, 1, 1, 1)
    forbidden = np.zeros(shape, dtype=bool)
    forbidden[:, :, 0, 0, 0, 0] = mask_2d

    input_path.parent.mkdir(parents=True, exist_ok=True)
    coords = [idx.tolist() for idx in np.argwhere(forbidden)]
    payload = {"shape": list(shape), "forbidden": coords}
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    ret = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--figure",
            str(figure_path),
            "--n-bootstrap",
            "32",
            "--seed",
            "7",
        ]
    )
    assert ret == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["n_boundary"] >= 100
    assert np.isfinite(result["H"])
    assert np.isfinite(result["ci"][0])
    assert np.isfinite(result["ci"][1])
    assert np.isfinite(result["r2"])
    assert figure_path.is_file()
