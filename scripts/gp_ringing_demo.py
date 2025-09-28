"""Multi-frequency GP ringing scaffolds.

This module extends the synthetic ringing demo with helper utilities for
band-specific filtering, lightweight GP feature extraction, and plotting.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Mapping, MutableMapping

import matplotlib
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from .gp_freq_bands import FREQUENCY_BANDS

matplotlib.use("Agg")  # Headless-friendly backend
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.join("results", "gp_demo", "multi_frequency_results.json")
FIGURE_PATH = os.path.join("figures", "multi_freq_lambda_star.png")


def bandpass_filter(
    data: np.ndarray,
    f_low: float,
    f_high: float,
    fs: float = 250.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""

    if f_low <= 0 or f_high >= fs / 2:
        raise ValueError("Band edges must be within (0, fs/2).")
    if f_low >= f_high:
        raise ValueError("f_low must be < f_high")

    nyq = fs / 2.0
    low = f_low / nyq
    high = f_high / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def _resample_lambda(lambda_schedule: np.ndarray, target_length: int) -> np.ndarray:
    lam = np.asarray(lambda_schedule, dtype=float)
    if lam.size == target_length:
        return lam
    x_old = np.linspace(0.0, 1.0, lam.size)
    x_new = np.linspace(0.0, 1.0, target_length)
    return np.interp(x_new, x_old, lam)


def gp_analysis_pipeline(filtered_data: np.ndarray, lambda_schedule: np.ndarray) -> Dict[str, float]:
    """Placeholder GP analysis hook.

    The real implementation should plug into the GP-specific feature extractors.
    For now we expose lightweight envelope-based summaries that keep the stub
    deterministic and fast for smoke tests.
    """

    data = np.asarray(filtered_data, dtype=float)
    lam = _resample_lambda(lambda_schedule, data.size)
    if data.ndim != 1:
        raise ValueError("Filtered data must be 1-D")

    envelope = np.abs(hilbert(data))
    if not np.any(np.isfinite(envelope)):
        raise ValueError("Envelope computation failed")

    lambda_star_idx = int(np.argmax(envelope))
    lambda_star = float(lam[lambda_star_idx])
    mi_peak = float(np.max(envelope))
    gradient = np.gradient(envelope)
    transition_sharpness = float(np.max(np.abs(gradient)))
    hysteresis_area = float(np.trapezoid(np.abs(envelope - envelope.mean()), lam))

    return {
        "lambda_star": lambda_star,
        "mi_peak": mi_peak,
        "transition_sharpness": transition_sharpness,
        "hysteresis_area": hysteresis_area,
        "envelope_mean": float(np.mean(envelope)),
        "envelope_std": float(np.std(envelope)),
    }


def _aggregate_metrics(per_band: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    keys = ["lambda_star", "hysteresis_area", "mi_peak", "transition_sharpness"]
    aggregate = {}
    for key in keys:
        values = [metrics[key] for metrics in per_band.values() if key in metrics]
        aggregate[key] = float(np.mean(values)) if values else float("nan")
    return aggregate


def multi_band_gp_analysis(
    timeseries: np.ndarray,
    lambda_schedule: np.ndarray,
    fs: float = 250.0,
    bands: Mapping[str, tuple] = FREQUENCY_BANDS,
    save_path: str | None = RESULTS_PATH,
) -> Dict[str, object]:
    """Run the GP analysis stub across multiple frequency bands."""

    timeseries = np.asarray(timeseries, dtype=float)
    if timeseries.ndim != 1:
        raise ValueError("timeseries must be 1-D")

    per_band_results: MutableMapping[str, Dict[str, object]] = {}
    for name, (f_low, f_high) in bands.items():
        try:
            filtered = bandpass_filter(timeseries, f_low, f_high, fs=fs)
        except ValueError:
            # Skip invalid bands for the chosen sampling rate
            continue
        metrics = gp_analysis_pipeline(filtered, lambda_schedule)
        per_band_results[name] = {
            "f_low": float(f_low),
            "f_high": float(f_high),
            "metrics": metrics,
            "raw": filtered.tolist(),
        }

    aggregate = _aggregate_metrics({k: v["metrics"] for k, v in per_band_results.items()})
    results: Dict[str, object] = {
        **aggregate,
        "bands": per_band_results,
        "fs": float(fs),
        "lambda_schedule_length": int(np.asarray(lambda_schedule).size),
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def plot_lambda_star(results: Mapping[str, object], out_path: str = FIGURE_PATH) -> None:
    """Bar plot of lambda* per band."""

    bands = results.get("bands", {})
    if not bands:
        return

    names = list(bands.keys())
    values = [bands[name]["metrics"].get("lambda_star", np.nan) for name in names]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(names, values, color="#5696ff")
    plt.ylabel("λ*")
    plt.title("GP λ* across frequency bands")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _demo(lambda_schedule: np.ndarray | None = None) -> Dict[str, object]:
    fs = 250.0
    t = np.arange(0, 10.0, 1.0 / fs)
    alpha = np.sin(2 * np.pi * 10.0 * t)
    beta = 0.6 * np.sin(2 * np.pi * 20.0 * t)
    noise = 0.3 * np.random.default_rng(7).standard_normal(t.size)
    timeseries = alpha + beta + noise

    if lambda_schedule is None:
        lambda_schedule = np.linspace(0.0, 1.0, timeseries.size)

    results = multi_band_gp_analysis(timeseries, lambda_schedule, fs=fs)
    plot_lambda_star(results)
    return results


def main() -> None:
    results = _demo()
    print(f"[OK] Saved multi-band results to {RESULTS_PATH}")
    print(f"[OK] Saved λ* figure to {FIGURE_PATH}")
    print(json.dumps({k: results[k] for k in ("lambda_star", "mi_peak")}, indent=2))


if __name__ == "__main__":
    main()
