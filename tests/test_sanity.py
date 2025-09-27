"""Lightweight environment sanity checks for CI."""

from __future__ import annotations

import importlib
import sys

import numpy as np


def test_python_version() -> None:
    """Ensure we are running on a modern Python that supports the project."""

    assert sys.version_info >= (3, 10)


def test_numpy_rng_reproducible() -> None:
    """Check that numpy's default RNG is available and deterministic."""

    rng = np.random.default_rng(0)
    sample = rng.integers(0, 100, size=4)
    assert sample.tolist() == [85, 63, 51, 26]


def test_core_modules_importable() -> None:
    """The key modules should import without raising."""

    for name in [
        "spin_foam_mc_optimized",
        "experiments.spin_foam_smoke",
    ]:
        module = importlib.import_module(name)
        assert module is not None
