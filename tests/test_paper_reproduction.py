"""Re-derives the central claim of docs/papers/hallucination/.

The paper boxes a prediction for the grounded->creative boundary:

    eta * Ibar ~= lambda + gamma

i.e. a line of slope 1 with intercept gamma (0.5 at the paper's settings).
This test runs the repo's own phase sweep and asserts the fitted line still
has that shape. If someone changes the dynamics in phase_dynamics.py, the
paper's headline result stops being true and CI says so on that push --
rather than a reader discovering it later.

Deliberately NOT asserted here: the specific coefficients printed in the
paper's section 4.1 (0.346*lambda + 0.506). Those are not reproducible from
anything in this repository; see docs/papers/hallucination/REPRODUCTION.md.
The bounds below bracket the THEORY, which is what the paper actually claims.
"""

import importlib.util
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "hallucination" / "ci_boundary_check.yaml"
SCRIPT = ROOT / "experiments" / "hallucination" / "run_phase_map.py"


def _load_phase_map_module():
    spec = importlib.util.spec_from_file_location("run_phase_map", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_phase_boundary_matches_boxed_prediction():
    module = _load_phase_map_module()
    config = yaml.safe_load(CONFIG.read_text())

    swept = module.run_phase_sweep(config)
    results = swept[0] if isinstance(swept, tuple) else swept

    slope, intercept, points = module.compute_boundary_fit(results)

    assert slope is not None, "no boundary points found; the sweep found no |lambda_max| < 0.1"
    assert len(points) >= 3, f"only {len(points)} boundary points; fit is not meaningful"

    # Predicted: slope 1.0, intercept gamma.
    gamma = config["gamma"]
    assert 0.85 <= slope <= 1.20, (
        f"boundary slope {slope:.3f} no longer matches the paper's prediction of 1.0"
    )
    assert abs(intercept - gamma) <= 0.15, (
        f"boundary intercept {intercept:.3f} no longer matches gamma={gamma}"
    )
