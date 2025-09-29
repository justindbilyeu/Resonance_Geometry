"""Utilities for loading resonance mapper data formats."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BandResult:
    band_id: str
    frequency_range: List[float]
    lambda_star: float
    hysteresis_area: float
    transition_sharpness: float
    mi_peaks: List[Dict[str, float]]
    trajectory: List[float]
    betti_trace: List[int]
    notes: Optional[str] = None


@dataclass
class MultiFreqResults:
    metadata: Dict[str, Any]
    bands: List[BandResult]
    cross_band: Dict[str, Dict[str, float]]


def load_multi_freq(path: str) -> MultiFreqResults:
    """Load the multi-frequency results JSON specification."""
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    bands = [BandResult(**band) for band in payload.get("bands", [])]
    return MultiFreqResults(
        metadata=payload.get("metadata", {}),
        bands=bands,
        cross_band=payload.get("cross_band", {}),
    )


def load_spin_foam(path: str) -> Dict:
    """Load spin foam data saved as ``.npy`` or JSON."""
    if path.endswith((".json", ".JSON")):
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return np.load(path, allow_pickle=True).item()


def load_microtubule(path: str) -> Dict:
    """Load microtubule coherence traces saved as ``.npy`` or JSON."""
    if path.endswith((".json", ".JSON")):
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return np.load(path, allow_pickle=True).item()
