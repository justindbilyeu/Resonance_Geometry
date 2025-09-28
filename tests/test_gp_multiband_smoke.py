import json
from pathlib import Path

import numpy as np

from scripts.gp_freq_bands import FREQUENCY_BANDS
from scripts.gp_ringing_demo import RESULTS_PATH, multi_band_gp_analysis


def test_multi_band_smoke(tmp_path):
    fs = 250.0
    duration = 10.0
    t = np.arange(0, duration, 1.0 / fs)
    alpha = np.sin(2 * np.pi * 10.0 * t)
    beta = 0.5 * np.sin(2 * np.pi * 20.0 * t)
    noise = 0.2 * np.random.default_rng(123).standard_normal(t.size)
    timeseries = alpha + beta + noise

    lambda_schedule = np.linspace(0.0, 1.0, timeseries.size)

    results_path = Path(RESULTS_PATH)
    if results_path.exists():
        results_path.unlink()

    results = multi_band_gp_analysis(timeseries, lambda_schedule, fs=fs, bands=FREQUENCY_BANDS)

    assert results_path.exists(), "Expected multi-frequency results JSON to be written"
    assert "bands" in results
    assert "alpha" in results["bands"]
    assert "beta" in results["bands"]
    assert "lambda_star" in results
    assert "hysteresis_area" in results

    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["bands"]["alpha"]["metrics"].keys() >= {"lambda_star", "mi_peak"}
    assert payload["bands"]["beta"]["metrics"].keys() >= {"lambda_star", "mi_peak"}
