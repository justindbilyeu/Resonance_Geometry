"""Driver for running multi-frequency GP analysis and exporting JSON results."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch


def _ensure_repo_on_path() -> None:
    if __package__:
        return
    repo_root = Path(__file__).resolve().parents[1]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_repo_on_path()

try:
    from experiments.gp_ringing_demo import simulate_coupled, windowed_mi
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Unable to import GP ringing demo utilities. "
        "Ensure the repository is installed as a package or run from repo root."
    ) from exc


logger = logging.getLogger(__name__)

FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 100.0),
}


@dataclass
class BandMetrics:
    band_id: str
    frequency_range: Tuple[float, float]
    lambda_star: float
    hysteresis_area: float
    transition_sharpness: float
    mi_peaks: List[Dict[str, float]]
    trajectory: List[float]
    betti_trace: List[int]


def _band_power(values: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    if len(values) == 0:
        return 0.0
    nperseg = min(len(values), 256)
    noverlap = min(128, max(0, nperseg // 2))
    freqs, psd = welch(values, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _aggregate_by_lambda(
    lam_series: np.ndarray, value_series: np.ndarray, nbins: int
) -> Tuple[np.ndarray, np.ndarray]:
    if nbins < 2:
        raise ValueError("Number of bins must be at least 2 to aggregate trajectory")

    lam_min, lam_max = float(np.nanmin(lam_series)), float(np.nanmax(lam_series))
    bins = np.linspace(lam_min, lam_max, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    aggregated = np.full_like(centers, fill_value=np.nan, dtype=float)

    for i in range(nbins):
        mask = (lam_series >= bins[i]) & (lam_series < bins[i + 1])
        if np.any(mask):
            aggregated[i] = float(np.nanmean(value_series[mask]))
    return centers, aggregated


def _fill_nan_interp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return np.zeros_like(y)
    if mask.sum() == 1:
        filled = np.zeros_like(y)
        filled[:] = y[mask][0]
        return filled
    return np.interp(x, x[mask], y[mask])


def _extract_peaks(x: np.ndarray, y: np.ndarray, max_peaks: int = 5) -> List[Dict[str, float]]:
    peaks: List[Tuple[float, float]] = []
    for idx in range(1, len(y) - 1):
        if y[idx] >= y[idx - 1] and y[idx] >= y[idx + 1]:
            peaks.append((float(y[idx]), float(x[idx])))
    peaks.sort(key=lambda item: item[0], reverse=True)
    top_peaks = peaks[:max_peaks]
    return [{"value": val, "position": pos} for val, pos in top_peaks]


def _compute_hysteresis(
    lam_series: np.ndarray,
    value_series: np.ndarray,
    nbins: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    lam_max_idx = int(np.argmax(lam_series))
    up_mask = np.arange(len(lam_series)) <= lam_max_idx
    down_mask = ~up_mask

    centers, up_vals = _aggregate_by_lambda(lam_series[up_mask], value_series[up_mask], nbins)
    _, down_vals = _aggregate_by_lambda(
        lam_series[down_mask], value_series[down_mask], nbins
    )

    up_filled = _fill_nan_interp(centers, up_vals)
    down_filled = _fill_nan_interp(centers, down_vals)
    area = float(np.trapezoid(np.abs(up_filled - down_filled), centers))
    return area, centers, up_filled, down_filled


def _compute_transition_sharpness(lam_centers: np.ndarray, trajectory: np.ndarray) -> float:
    if len(lam_centers) < 2:
        return 0.0
    gradient = np.gradient(trajectory, lam_centers, edge_order=1)
    return float(np.nanmax(np.abs(gradient)))


def _sample_lambda_schedule(lam_series: np.ndarray, count: int) -> List[float]:
    if len(lam_series) == 0:
        return []
    unique = np.unique(lam_series)
    count = max(2, min(count, len(unique)))
    indices = np.linspace(0, len(unique) - 1, num=count, dtype=int)
    return unique[indices].astype(float).tolist()


def _bandpass_filter(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    if len(signal) == 0:
        return signal
    nyq = 0.5 * fs
    low, high = band
    low = max(0.0, low)
    high = min(nyq - 1e-6, high)
    if high <= 0 or low >= high:
        return signal
    wn = [low / nyq, high / nyq]
    order = 4
    b, a = butter(order, wn, btype="bandpass")
    return filtfilt(b, a, signal)


def _compute_band_metrics(
    band_id: str,
    freq_range: Tuple[float, float],
    lam_series: np.ndarray,
    mi_series_collection: Sequence[np.ndarray],
    nbins: int,
    fs_signal: float,
    mi_hop: int,
) -> BandMetrics:
    mi_matrix = np.vstack(mi_series_collection)
    mi_avg = np.nanmean(mi_matrix, axis=0)

    area, lam_centers, up_curve, down_curve = _compute_hysteresis(
        lam_series, mi_avg, nbins
    )

    combined_curve = 0.5 * (up_curve + down_curve)
    lambda_star = float(lam_centers[int(np.argmax(combined_curve))])

    filled_curve = _fill_nan_interp(lam_centers, combined_curve)
    transition_sharpness = _compute_transition_sharpness(lam_centers, filled_curve)

    mi_peaks = _extract_peaks(lam_centers, filled_curve)

    fs_mi = fs_signal / mi_hop if mi_hop > 0 else fs_signal
    band_power = _band_power(mi_avg - np.mean(mi_avg), fs=fs_mi, band=freq_range)
    logger.info(
        "Band %s → λ*: %.4f, hysteresis area: %.4f, band power: %.4f",
        band_id,
        lambda_star,
        area,
        band_power,
    )

    trajectory = filled_curve.tolist()
    threshold = float(np.nanmedian(filled_curve)) if len(filled_curve) else 0.0
    betti_trace = [int(val > threshold) for val in filled_curve]

    return BandMetrics(
        band_id=band_id,
        frequency_range=freq_range,
        lambda_star=lambda_star,
        hysteresis_area=area,
        transition_sharpness=transition_sharpness,
        mi_peaks=mi_peaks,
        trajectory=trajectory,
        betti_trace=betti_trace,
    )


def run_pipeline(
    steps: int,
    reps: int,
    output_dir: str,
    base_seed: int = 7,
    fs: float = 128.0,
    dur_up: int = 60,
    dur_down: int = 60,
    lam_max: float = 0.9,
    win: int = 256,
    hop: int = 64,
) -> Dict[str, object]:
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        "Running multi-frequency GP pipeline with %d steps, %d reps → %s",
        steps,
        reps,
        output_dir,
    )

    mi_results: Dict[str, BandMetrics] = {}
    lambda_windows: np.ndarray | None = None
    nbins = max(steps, 10)

    for band_id, freq_range in FREQUENCY_BANDS.items():
        mi_series_collection: List[np.ndarray] = []
        for rep in range(reps):
            seed = base_seed + rep
            lam, x, y = simulate_coupled(
                fs=int(fs),
                dur_up=dur_up,
                dur_dn=dur_down,
                lam_max=lam_max,
                seed=seed,
            )
            x_band = _bandpass_filter(x, fs, freq_range)
            y_band = _bandpass_filter(y, fs, freq_range)
            starts, mi_vals = windowed_mi(x_band, y_band, win=win, hop=hop, bins=64)
            if lambda_windows is None:
                midpoints = starts + win // 2
                lambda_windows = lam[midpoints].astype(float)
            mi_series_collection.append(mi_vals)
        if lambda_windows is None:
            raise RuntimeError("Failed to compute lambda windows for MI series")

        metrics = _compute_band_metrics(
            band_id,
            freq_range,
            lambda_windows,
            mi_series_collection,
            nbins=nbins,
            fs_signal=fs,
            mi_hop=hop,
        )
        mi_results[band_id] = metrics

    lambda_schedule = _sample_lambda_schedule(lambda_windows, nbins)

    if lambda_windows is None:
        raise RuntimeError("Lambda schedule could not be derived from simulations")

    metadata = {
        "sim_id": "multi_freq_gp_v1",
        "source": "synthetic",
        "fs": fs,
        "lambda_schedule": lambda_schedule,
        "timestamp": datetime.now(UTC).isoformat(),
        "notes": (
            "Trajectory curves derived from averaged mutual information across "
            f"{reps} repetitions and aggregated into {nbins} λ bins."
        ),
    }

    bands_payload = [
        {
            "band_id": metrics.band_id,
            "frequency_range": list(metrics.frequency_range),
            "lambda_star": metrics.lambda_star,
            "hysteresis_area": metrics.hysteresis_area,
            "transition_sharpness": metrics.transition_sharpness,
            "mi_peaks": metrics.mi_peaks,
            "trajectory": metrics.trajectory,
            "betti_trace": metrics.betti_trace,
        }
        for metrics in mi_results.values()
    ]

    payload = {
        "metadata": metadata,
        "bands": bands_payload,
        "cross_band": {},
    }

    output_path = os.path.join(output_dir, "multi_frequency_results.json")
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    logger.info("Wrote multi-frequency results → %s", output_path)
    return payload


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-frequency GP driver")
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of λ bins used when aggregating trajectories (default: 200)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of stochastic repetitions of the GP simulation (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "gp_demo"),
        help="Directory where the JSON payload will be saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base RNG seed used for the simulations",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        run_pipeline(steps=args.steps, reps=args.reps, output_dir=args.output_dir, base_seed=args.seed)
    except Exception as exc:  # pragma: no cover - CLI protection
        logger.error("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
