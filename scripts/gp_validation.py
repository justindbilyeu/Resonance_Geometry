"""Validation scaffolding for the GP multi-frequency demo."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.signal import butter, filtfilt

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from scripts.gp_freq_bands import FREQUENCY_BANDS
    from scripts.gp_ringing_demo import multi_band_gp_analysis
else:
    from .gp_freq_bands import FREQUENCY_BANDS
    from .gp_ringing_demo import multi_band_gp_analysis

RESULTS_PATH = os.path.join("results", "gp_demo", "multi_frequency_validation.json")


def _bandpass_noise(noise: np.ndarray, band: Sequence[float], fs: float) -> np.ndarray:
    low, high = band
    nyq = fs / 2.0
    if low <= 0 or high >= nyq or low >= high:
        raise ValueError("Invalid band specification")
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, noise)


def generate_phase_randomized_surrogate(timeseries: Sequence[float], rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    data = np.asarray(timeseries, dtype=float)
    spectrum = np.fft.rfft(data)
    magnitudes = np.abs(spectrum)
    phases = rng.uniform(0, 2 * np.pi, size=spectrum.shape)
    phases[0] = 0.0
    if data.size % 2 == 0:
        phases[-1] = 0.0
    surrogate_spec = magnitudes * np.exp(1j * phases)
    surrogate = np.fft.irfft(surrogate_spec, n=data.size)
    return surrogate.astype(float)


def generate_filtered_noise_surrogates(
    timeseries: Sequence[float],
    bands: Mapping[str, Sequence[float]] | None = None,
    fs: float = 250.0,
    n_surrogates: int = 10,
    rng: np.random.Generator | None = None,
) -> List[np.ndarray]:
    rng = rng or np.random.default_rng()
    data = np.asarray(timeseries, dtype=float)
    bands = bands or FREQUENCY_BANDS
    band_values = list(bands.values())
    surrogates: List[np.ndarray] = []
    for _ in range(n_surrogates):
        noise = rng.standard_normal(size=data.size)
        band = band_values[int(rng.integers(0, len(band_values)))]
        filtered = _bandpass_noise(noise, band, fs=fs)
        surrogates.append(filtered)
    return surrogates


def generate_time_reversed_surrogates(timeseries: Sequence[float]) -> np.ndarray:
    data = np.asarray(timeseries, dtype=float)
    return data[::-1].copy()


def generate_parameter_shuffled_surrogates(
    real_results: Mapping[str, object],
    rng: np.random.Generator | None = None,
) -> Dict[str, List[float]]:
    rng = rng or np.random.default_rng()
    bands = real_results.get("bands", {})
    collected: Dict[str, List[float]] = {"lambda_star": [], "mi_peak": [], "hysteresis_area": [], "transition_sharpness": []}
    if not bands:
        return collected
    for metrics_key in collected:
        values = [band_data["metrics"].get(metrics_key, np.nan) for band_data in bands.values()]
        rng.shuffle(values)
        collected[metrics_key] = [float(v) for v in values]
    return collected


def apply_multiple_comparisons_correction(p_values: Sequence[float], method: str = "fdr_bh") -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if method == "bonferroni":
        return np.minimum(p_values * p_values.size, 1.0)
    if method != "fdr_bh":
        raise ValueError("Unsupported correction method")
    order = np.argsort(p_values)
    ranked = np.empty_like(p_values)
    m = p_values.size
    for i, idx in enumerate(order, start=1):
        ranked[idx] = p_values[idx] * m / i
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(adjusted, 0.0, 1.0)


def compute_effect_sizes(real_results: Mapping[str, object], surrogate_results: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    metrics = ["lambda_star", "mi_peak", "hysteresis_area", "transition_sharpness"]
    effects: Dict[str, float] = {}
    for metric in metrics:
        real_value = float(real_results.get(metric, np.nan))
        surrogate_vals = [float(s.get(metric, np.nan)) for s in surrogate_results]
        surrogate_mean = float(np.nanmean(surrogate_vals)) if surrogate_vals else float("nan")
        surrogate_std = float(np.nanstd(surrogate_vals)) if surrogate_vals else float("nan")
        if np.isnan(surrogate_std) or surrogate_std == 0:
            effects[metric] = float("nan")
        else:
            effects[metric] = (real_value - surrogate_mean) / surrogate_std
    return effects


def run_full_validation_suite(
    timeseries: Sequence[float],
    lambda_schedule: Sequence[float],
    real_results: Mapping[str, object],
    fs: float = 250.0,
    bands: Mapping[str, Sequence[float]] | None = None,
    n_surrogates: int = 10,
) -> Dict[str, object]:
    bands = bands or FREQUENCY_BANDS
    rng = np.random.default_rng(123)

    phase_surrogates = [generate_phase_randomized_surrogate(timeseries, rng=rng) for _ in range(n_surrogates)]
    filtered_surrogates = generate_filtered_noise_surrogates(timeseries, bands=bands, fs=fs, n_surrogates=n_surrogates, rng=rng)
    reversed_surrogate = generate_time_reversed_surrogates(timeseries)

    surrogate_results = []
    for surrogate in phase_surrogates + filtered_surrogates + [reversed_surrogate]:
        surrogate_results.append(
            multi_band_gp_analysis(np.asarray(surrogate), lambda_schedule, fs=fs, bands=bands, save_path=None)
        )

    parameter_shuffle = generate_parameter_shuffled_surrogates(real_results, rng=rng)
    effects = compute_effect_sizes(real_results, surrogate_results)

    summary = {
        "phase_surrogate_count": len(phase_surrogates),
        "filtered_surrogate_count": len(filtered_surrogates),
        "parameter_shuffle": parameter_shuffle,
        "effect_sizes": effects,
    }
    return summary


def generate_validation_summary(payload: Mapping[str, object]) -> Dict[str, object]:
    effects = payload.get("effect_sizes", {})
    significant = {k: v for k, v in effects.items() if np.isfinite(v) and abs(v) > 1.0}
    return {
        "significant_effects": significant,
        "effect_sizes": effects,
        "notes": "Thresholded at |z| > 1.0 for quick-look significance.",
    }


def _demo_validation() -> Dict[str, object]:
    fs = 250.0
    t = np.arange(0, 10.0, 1.0 / fs)
    alpha = np.sin(2 * np.pi * 10.0 * t)
    beta = 0.6 * np.sin(2 * np.pi * 20.0 * t)
    noise = 0.3 * np.random.default_rng(1).standard_normal(t.size)
    timeseries = alpha + beta + noise
    lambda_schedule = np.linspace(0.0, 1.0, timeseries.size)

    real_results = multi_band_gp_analysis(timeseries, lambda_schedule, fs=fs, bands=FREQUENCY_BANDS, save_path=None)
    payload = run_full_validation_suite(timeseries, lambda_schedule, real_results, fs=fs, bands=FREQUENCY_BANDS)
    summary = generate_validation_summary(payload)

    results = {"payload": payload, "summary": summary}
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    results = _demo_validation()
    print(f"[OK] Saved validation summary to {RESULTS_PATH}")
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
