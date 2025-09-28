"""Frequency band definitions used by GP multi-band demos."""

from __future__ import annotations

FREQUENCY_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}

GP_BANDS = {
    "core": (4.0, 40.0),
    "extended": (1.0, 80.0),
}

__all__ = ["FREQUENCY_BANDS", "GP_BANDS"]
