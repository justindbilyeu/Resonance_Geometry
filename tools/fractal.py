"""Utility routines for estimating fractal dimension via box counting."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = ["BoxCountStats", "boxcount_dimension"]


@dataclass(slots=True)
class BoxCountStats:
    """Container describing a fitted box-counting curve."""

    scales: np.ndarray
    counts: np.ndarray
    slope: float
    intercept: float
    r2: float
    ci: tuple[float, float]


def _prepare_points(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("boxcount_dimension expects a 2D binary mask")
    if mask.size == 0:
        return np.zeros((0, 2), dtype=int)
    coords = np.argwhere(mask.astype(bool))
    return coords.astype(float, copy=False)


def _choose_scales(points: np.ndarray, n_scales: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.asarray([], dtype=float)
    span = points.max(axis=0) - points.min(axis=0) + 1.0
    max_extent = float(np.max(span))
    max_extent = max(max_extent, 2.0)
    raw = np.logspace(0.0, math.log10(max_extent), num=n_scales)
    scales = np.unique(np.clip(np.floor(raw + 1e-8).astype(int), 1, None))
    return scales.astype(float)


def _box_counts(points: np.ndarray, scales: Iterable[float]) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(len(list(scales)), dtype=float)
    mins = points.min(axis=0)
    shifted = points - mins
    counts = []
    for scale in scales:
        step = max(int(round(scale)), 1)
        bins = np.floor_divide(shifted, step)
        unique = np.unique(bins, axis=0)
        counts.append(float(unique.shape[0]))
    return np.asarray(counts, dtype=float)


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def boxcount_dimension(
    mask: np.ndarray,
    *,
    n_scales: int = 12,
    n_boot: int = 200,
    seed: int | None = None,
) -> dict[str, object]:
    """Estimate the fractal dimension of a 2D binary mask.

    Parameters
    ----------
    mask:
        Boolean or numeric array where non-zero entries indicate boundary pixels.
    n_scales:
        Number of geometric scales used for the box-counting regression.
    n_boot:
        Number of bootstrap resamples used to derive a 95% confidence interval.
    seed:
        Optional RNG seed to make the bootstrap deterministic.
    """

    points = _prepare_points(np.asarray(mask))
    if points.shape[0] < 2:
        return {"H": float("nan"), "r2": float("nan"), "ci": (float("nan"), float("nan"))}

    scales = _choose_scales(points, n_scales)
    if scales.size < 2:
        return {"H": float("nan"), "r2": float("nan"), "ci": (float("nan"), float("nan"))}

    counts = _box_counts(points, scales)
    valid = counts > 0
    if valid.sum() < 2:
        return {"H": float("nan"), "r2": float("nan"), "ci": (float("nan"), float("nan"))}

    counts = counts[valid]
    scales = scales[valid]

    log_counts = np.log(counts)
    log_inv_scale = np.log(1.0 / scales)
    slope, intercept, r2 = _fit_line(log_inv_scale, log_counts)

    rng = np.random.default_rng(seed)
    n_points = points.shape[0]
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n_points, size=n_points)
        sample_points = points[sample_idx]
        sample_counts = _box_counts(sample_points, scales)
        if np.any(sample_counts <= 0):
            boot[i] = float("nan")
            continue
        log_counts_bs = np.log(sample_counts)
        slope_bs, _, _ = _fit_line(log_inv_scale, log_counts_bs)
        boot[i] = slope_bs

    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        ci = (float("nan"), float("nan"))
    else:
        ci = tuple(np.percentile(boot, [2.5, 97.5]).astype(float))  # type: ignore[assignment]

    return {"H": slope, "r2": r2, "ci": ci, "scales": scales.tolist(), "counts": counts.tolist()}
