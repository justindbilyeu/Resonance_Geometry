#!/usr/bin/env python3
"""Lightweight placeholder simulation harness.

This module provides a deterministic approximation of the dynamics that the
real meta flow simulator would normally compute.  It is intentionally simple
but keeps the public API the rest of the project expects so that build and CI
pipelines can be exercised without the heavy numerical dependency stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass
class Trajectory:
    """Container mirroring the dictionary interface used downstream."""

    t: np.ndarray
    lambda_max: np.ndarray
    regime: int

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def _classify_regime(lam: float, eta: float, gamma: float) -> int:
    """Toy classifier for grounded/creative/hallucinatory regimes."""
    boundary = lam + gamma
    if eta > boundary:
        return 2  # hallucinatory
    if eta > lam:
        return 1  # marginal/creative
    return 0  # grounded


def simulate_trajectory(params: Dict[str, float], T_max: float = 3.0, dt: float = 0.01) -> Trajectory:
    """Return a lightweight trajectory with qualitative behaviour.

    The function mirrors the expected API of the original simulator so the
    validation scripts can run in environments where the heavy simulator is not
    available.  We synthesise a smooth signal for ``lambda_max`` that depends on
    ``eta`` and ``lambda`` in a way that mimics the intended phase structure.
    """

    lam = float(params.get('lambda', 1.0))
    eta = float(params.get('eta', 1.0))
    gamma = float(params.get('gamma', 0.5))
    k = float(params.get('k', 1.0))

    t = np.arange(0.0, T_max + 0.5 * dt, dt)
    envelope = np.tanh(k * t)
    offset = eta - lam - gamma / 2.0
    lambda_max = envelope * offset

    regime = _classify_regime(lam, eta, gamma)

    return Trajectory(t=t, lambda_max=lambda_max, regime=regime)


def batch_simulate(grid: Iterable[Dict[str, float]], **kwargs) -> Iterable[Trajectory]:
    """Helper to simulate a collection of parameter dictionaries."""
    for params in grid:
        yield simulate_trajectory(params, **kwargs)
