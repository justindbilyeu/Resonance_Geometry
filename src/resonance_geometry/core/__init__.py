"""Resonance Geometry core package.

Contains fundamental components:
- state_vector: State vector dynamics
- forbidden: Forbidden region analysis
- fractals: Fractal dimension computations
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

# Export core modules for convenient access
from . import state_vector, forbidden, fractals

__all__ = ["state_vector", "forbidden", "fractals"]
