"""
Resonance Geometry: Toy Universe (v2.1 Canon)
----------------------------------------------
A foundational physics engine combining Kuramoto dynamics with Geometric Plasticity.

This module justifies the ITPU (Information Throughput Potential) metrics
through rigorous simulation of adaptive coupling on a resonant substrate.

Components:
- ResonanceUniverse: The core simulation engine
- science_suite: Automated parameter sweep tools
"""

from .resonance_universe import ResonanceUniverse
from .science_suite import measure_performance, run_alpha_beta_sweep

__version__ = "2.1.0"
__all__ = ["ResonanceUniverse", "measure_performance", "run_alpha_beta_sweep"]
