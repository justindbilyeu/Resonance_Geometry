import json
import os

import pytest

networkx = pytest.importorskip("networkx")
import numpy as np

from tools.resonance_mapper.graphify import build_graph_from_multi_freq
from tools.resonance_mapper.gnn_encoder import train_gnn
from tools.resonance_mapper.loader import load_multi_freq
from tools.resonance_mapper.tda import compute_tda


def _fake_payload():
    return {
        "metadata": {"sim_id": "smoke"},
        "bands": [
            {
                "band_id": "alpha",
                "frequency_range": [8.0, 12.0],
                "lambda_star": 1.2,
                "hysteresis_area": 0.4,
                "transition_sharpness": 0.3,
                "mi_peaks": [{"value": 0.2, "position": 1.0}],
                "trajectory": [0.1, 0.2, 0.3],
                "betti_trace": [1, 1, 2],
                "notes": None,
            }
        ],
        "cross_band": {},
    }


def test_mapper_smoke(tmp_path):
    payload = _fake_payload()
    temp_json = tmp_path / "temp.json"
    temp_json.write_text(json.dumps(payload))
    results = load_multi_freq(temp_json)
    graph = build_graph_from_multi_freq(results)
    assert isinstance(graph, networkx.Graph)
    embeddings = train_gnn(graph, epochs=5)
    betti, _, _ = compute_tda(embeddings)
    assert len(betti) == 3
    output_dir = tmp_path / "results" / "mapper"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "tda_diagram.npy", np.zeros((1, 2)))
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump({"betti": betti}, handle)
    assert os.path.exists(output_dir / "report.json")
