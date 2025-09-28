# Resonance Mapper Addendum

## Ingestion Schema: `multi_frequency_results.json`

```json
{
  "metadata": {
    "sim_id": "gp_demo_v2",
    "source": "synthetic" | "empirical",
    "fs": 256,
    "lambda_schedule": [0.1, 0.2, "..."],
    "notes": "optional string"
  },
  "bands": [
    {
      "band_id": "alpha",
      "frequency_range": [8.0, 12.0],
      "lambda_star": 1.25,
      "hysteresis_area": 0.42,
      "transition_sharpness": 0.31,
      "mi_peaks": [
        {"value": 1.7, "position": 2.1}
      ],
      "trajectory": [0.1, 0.2, "..."],
      "betti_trace": [1, 1, 2],
      "notes": "optional string"
    }
  ],
  "cross_band": {
    "alpha_theta": {
      "coupling_strength": 0.18,
      "p_value": 0.03,
      "direction": "alpha→theta"
    }
  }
}
```

## Python Dataclasses & Loader Stub

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import json

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
    notes: str | None = None

@dataclass
class MultiFreqResults:
    metadata: Dict[str, Any]
    bands: List[BandResult]
    cross_band: Dict[str, Dict[str, float]]


def load_multi_freq_results(path: str) -> MultiFreqResults:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    bands = [BandResult(**band) for band in payload.get("bands", [])]
    return MultiFreqResults(
        metadata=payload.get("metadata", {}),
        bands=bands,
        cross_band=payload.get("cross_band", {})
    )
```

## Invariant Functions (Pseudocode)

```python
def betti_from_trajectories(trajectory_matrix):
    """Compute Betti numbers from delay-embedded trajectories (Mapper-compatible)."""


def persistence_diagram_from_band(band_result):
    """Return persistence pairs using Vietoris–Rips filtered on the band trajectory."""


def curvature_estimate_from_mapper(graph):
    """Estimate discrete curvature (e.g., Forman or Ollivier–Ricci) on Mapper graph nodes."""
```

- **Betti traces** derive from persistence landscapes sampled along the λ sweep.  
- **Persistence diagrams** inform Mapper feature selection; store barcodes per band for downstream comparison.  
- **Curvature estimates** capture negative curvature moats predicted by the Topological Constraint Test.

## Cross-Simulation Integration Plan

- Adopt a common graph representation: `nodes = [{"id": i, "features": {...}}]`, `edges = [(i, j, weight)]`.  
- Minimal invariant set: Betti (0,1), persistence entropy, curvature summary (mean, min), and hysteresis statistics.  
- Normalize metadata keys (`sim_id`, `source`, `fs`) for compatibility across synthetic and empirical pipelines.  
- Provide adapters for both GP demos and external datasets to emit the shared schema.

## Quick-Win Harness Outline

- Script: `tools/resonance_mapper_smoke.py`.  
- Generates a fake `multi_frequency_results.json` with minimal alpha-band payload.  
- Emits placeholder figures (`figures/mapper_smoke_persistence.png`) and summaries (`outputs/mapper_smoke_summary.json`).  
- Serves as a smoke test for Mapper ingestion, ensuring directories exist and JSON schema is respected.
