"""Utility script for running the multi-frequency GP demo pipeline.

This module stitches together the synthetic coupled-oscillator generator from
``experiments.gp_ringing_demo`` with the light-weight multi-band analysis
helpers that live under ``scripts/gp_ringing_demo``.  The original PR that
introduced the file wired the pieces together but the script subsequently went
missing, leaving downstream documentation and CLI instructions pointing at a
non-existent entry point.  The implementation below recreates the missing glue
with a small, well-tested API that can be imported or executed from the
command line.

The default configuration targets the synthetic demo that ships with the
repository.  It sweeps over a small range of maximum coupling strengths, runs
several random replicates, computes mutual information trajectories, and then
funnels the aggregated MI signal through the multi-band GP analysis stub.  The
results are serialised into ``results/gp_demo/multi_frequency`` so that the
existing README instructions regain their footing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np


def _ensure_repo_on_path() -> None:
    """Ensure the repository root is importable.

    When the script is executed directly (``__package__`` is empty) we insert
    the repository root onto ``sys.path`` so that local modules inside
    ``experiments`` and ``scripts`` can be imported without requiring
    installation as a package.
    """

    if __package__:
        return
    repo_root = Path(__file__).resolve().parents[1]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_repo_on_path()


try:  # pragma: no cover - import guard is exercised implicitly in tests.
    from experiments.gp_ringing_demo import (  # type: ignore
        alpha_power,
        simulate_coupled,
        windowed_mi,
    )
except ImportError as exc:  # pragma: no cover - highlighted in failure logs.
    raise ImportError(
        "Failed to import required helpers from experiments.gp_ringing_demo."
    ) from exc

try:  # pragma: no cover - optional dependency for smoke tests.
    from scripts.gp_freq_bands import FREQUENCY_BANDS  # type: ignore
    from scripts.gp_ringing_demo import multi_band_gp_analysis  # type: ignore
except ImportError:  # pragma: no cover - analysis layer is optional at runtime.
    FREQUENCY_BANDS = {}  # type: ignore[assignment]
    multi_band_gp_analysis = None  # type: ignore[assignment]


DEFAULT_OUTPUT_DIR = Path("results") / "gp_demo" / "multi_frequency"


@dataclass
class MUTrajectory:
    """Container for a single mutual-information trajectory."""

    starts: np.ndarray
    values: np.ndarray
    lambda_centres: np.ndarray
    sampling_rate: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "starts": self.starts.tolist(),
            "values": self.values.tolist(),
            "lambda_centres": self.lambda_centres.tolist(),
            "sampling_rate": float(self.sampling_rate),
        }


@dataclass
class TrialResult:
    """Summary for a single replicate of the synthetic experiment."""

    trial_id: str
    lam_max: float
    seed: int
    mi_alpha_power: float
    mi_trajectory: MUTrajectory
    band_results: Mapping[str, Any]

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mi_trajectory"] = self.mi_trajectory.to_json()
        payload["band_results"] = _maybe_convert_np(self.band_results)
        return payload


def _maybe_convert_np(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to Python lists/numbers."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _maybe_convert_np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_maybe_convert_np(v) for v in obj]
    return obj


def compute_mi_trajectory(
    signal_x: np.ndarray,
    signal_y: np.ndarray,
    coupling_schedule: np.ndarray,
    fs: float,
    window: int = 256,
    hop: int = 64,
    bins: int = 64,
) -> MUTrajectory:
    """Compute the windowed mutual-information trajectory for x/y."""

    starts, values = windowed_mi(signal_x, signal_y, win=window, hop=hop, bins=bins)
    if starts.size == 0:
        raise ValueError("Windowed MI produced an empty trajectory")

    centre_indices = starts + window // 2
    centre_indices = np.clip(centre_indices, 0, coupling_schedule.size - 1)
    lambda_centres = coupling_schedule[centre_indices]
    sampling_rate = fs / hop

    return MUTrajectory(starts=starts, values=values, lambda_centres=lambda_centres, sampling_rate=sampling_rate)


def _run_multi_band_analysis(
    trajectory: MUTrajectory,
    bands: Optional[Mapping[str, Iterable[float]]] = None,
) -> Mapping[str, Any]:
    """Run the optional multi-band GP analysis on the MI trajectory."""

    if multi_band_gp_analysis is None:
        return {}

    bands = bands or FREQUENCY_BANDS
    # The helper accepts arbitrary iterables, but we ensure NumPy arrays for
    # predictable dtype promotion.
    timeseries = np.asarray(trajectory.values, dtype=float)
    lambda_schedule = np.asarray(trajectory.lambda_centres, dtype=float)
    return multi_band_gp_analysis(
        timeseries,
        lambda_schedule,
        fs=float(trajectory.sampling_rate),
        bands=bands,
        save_path=None,
    )


def run_single_trial(
    trial_id: str,
    lam_max: float,
    seed: int,
    fs: float,
    window: int,
    hop: int,
    bins: int,
    bands: Optional[Mapping[str, Iterable[float]]],
) -> TrialResult:
    """Execute one replicate of the coupled oscillator simulation."""

    lam, x, y = simulate_coupled(fs=fs, lam_max=lam_max, seed=seed)
    trajectory = compute_mi_trajectory(x, y, lam, fs=fs, window=window, hop=hop, bins=bins)

    centred_mi = trajectory.values - np.mean(trajectory.values)
    mi_alpha = alpha_power(centred_mi, fs=trajectory.sampling_rate, band=(8.0, 12.0))

    band_results = _run_multi_band_analysis(trajectory, bands=bands)

    return TrialResult(
        trial_id=trial_id,
        lam_max=float(lam_max),
        seed=int(seed),
        mi_alpha_power=float(mi_alpha),
        mi_trajectory=trajectory,
        band_results=band_results,
    )


def _aggregate_trials(trials: Iterable[TrialResult]) -> Mapping[str, float]:
    metrics: MutableMapping[str, List[float]] = {
        "mi_alpha_power": [],
        "lambda_star": [],
        "hysteresis_area": [],
        "mi_peak": [],
        "transition_sharpness": [],
    }

    for trial in trials:
        metrics["mi_alpha_power"].append(trial.mi_alpha_power)
        results = trial.band_results
        metrics["lambda_star"].append(float(results.get("lambda_star", np.nan)))
        metrics["hysteresis_area"].append(float(results.get("hysteresis_area", np.nan)))
        metrics["mi_peak"].append(float(results.get("mi_peak", np.nan)))
        metrics["transition_sharpness"].append(float(results.get("transition_sharpness", np.nan)))

    summary: Dict[str, float] = {}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            summary[key] = float("nan")
            continue
        summary[key] = float(np.nanmean(arr))
    return summary


def run_pipeline(
    steps: int = 5,
    reps: int = 3,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    base_seed: int = 7,
    lam_max_min: float = 0.4,
    lam_max_max: float = 0.9,
    fs: float = 128.0,
    window: int = 256,
    hop: int = 64,
    bins: int = 64,
    bands: Optional[Mapping[str, Iterable[float]]] = None,
) -> Mapping[str, Any]:
    """Run the full multi-frequency pipeline and persist artefacts to disk."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if reps <= 0:
        raise ValueError("reps must be positive")
    if lam_max_min <= 0 or lam_max_max <= 0:
        raise ValueError("λ_max bounds must be positive")

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    lam_values = np.linspace(lam_max_min, lam_max_max, steps)
    trials: List[TrialResult] = []

    for step_idx, lam_max in enumerate(lam_values):
        for rep in range(reps):
            seed = base_seed + step_idx * reps + rep
            trial_id = f"lam{step_idx:02d}_rep{rep:02d}"
            logging.debug("Running trial %s with λ_max=%.3f seed=%d", trial_id, lam_max, seed)
            trial = run_single_trial(
                trial_id=trial_id,
                lam_max=float(lam_max),
                seed=seed,
                fs=fs,
                window=window,
                hop=hop,
                bins=bins,
                bands=bands,
            )
            trials.append(trial)

            trial_path = output_dir / f"{trial_id}.json"
            with trial_path.open("w", encoding="utf-8") as handle:
                json.dump(trial.to_json(), handle, indent=2)

    aggregate = _aggregate_trials(trials)
    metadata = {
        "steps": int(steps),
        "reps": int(reps),
        "fs": float(fs),
        "window": int(window),
        "hop": int(hop),
        "bins": int(bins),
        "lam_max_min": float(lam_max_min),
        "lam_max_max": float(lam_max_max),
        "base_seed": int(base_seed),
        "bands": list((bands or FREQUENCY_BANDS).keys()),
        "output_dir": str(output_dir),
    }

    payload: Dict[str, Any] = {
        "metadata": metadata,
        "aggregate": aggregate,
        "trials": [trial.to_json() for trial in trials],
    }

    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multi-frequency GP pipeline")
    parser.add_argument("--steps", type=int, default=5, help="Number of λ_max points to sample")
    parser.add_argument("--reps", type=int, default=3, help="Number of replicates per λ_max")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store JSON artefacts",
    )
    parser.add_argument("--base-seed", type=int, default=7, help="Base random seed")
    parser.add_argument("--lam-max-min", type=float, default=0.4, help="Minimum λ_max value")
    parser.add_argument("--lam-max-max", type=float, default=0.9, help="Maximum λ_max value")
    parser.add_argument("--fs", type=float, default=128.0, help="Sampling rate for the simulator")
    parser.add_argument("--window", type=int, default=256, help="Window size for MI computation")
    parser.add_argument("--hop", type=int, default=64, help="Hop size for MI computation")
    parser.add_argument("--bins", type=int, default=64, help="Histogram bins for MI computation")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Mapping[str, Any]:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    payload = run_pipeline(
        steps=args.steps,
        reps=args.reps,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        lam_max_min=args.lam_max_min,
        lam_max_max=args.lam_max_max,
        fs=args.fs,
        window=args.window,
        hop=args.hop,
        bins=args.bins,
    )

    logging.info("Saved summary to %s", args.output_dir / "run_summary.json")
    return payload


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

