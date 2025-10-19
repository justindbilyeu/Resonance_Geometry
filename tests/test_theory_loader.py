import json
from pathlib import Path
from tools.load_phase_surface import load_phase_surfaces

def test_loader_tolerates_missing(tmp_path: Path):
    d = tmp_path / "docs" / "data" / "theory"
    d.mkdir(parents=True, exist_ok=True)
    # no files present
    res = load_phase_surfaces(d)
    assert res["total_rows"] == 0
    assert isinstance(res["files_present"], list)

def test_loader_parses_minimal_schema(tmp_path: Path):
    d = tmp_path / "docs" / "data" / "theory"
    d.mkdir(parents=True, exist_ok=True)
    # write a tiny "balanced" sample
    sample = [
        {"phi": 0.0, "R_phi": 0.4, "lambda_phi": 0.632, "alpha": 1.0, "beta": 0.8, "alpha_over_beta": 1.25, "regime": "balanced"},
        {"phi": 1.0, "R_phi": 0.8, "lambda_phi": 1.0,   "alpha": 1.0, "beta": 0.8, "alpha_over_beta": 1.25, "regime": "balanced"},
    ]
    (d / "phase_surface_balanced.json").write_text(json.dumps(sample))
    res = load_phase_surfaces(d)
    assert res["total_rows"] == 2
    assert res["regimes"]["balanced"]["count"] == 2
    assert "phase_surface_balanced.json" in res["files_present"]

