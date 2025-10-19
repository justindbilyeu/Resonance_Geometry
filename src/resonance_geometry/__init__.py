"""
Resonance Geometry: Geometric approaches to information dynamics

This package provides tools for studying phase transitions in 
information-processing systems using differential geometry and 
dynamical systems theory.
"""

__version__ = "0.1.0"
__author__ = "Justin Bilyeu"

# Import main modules for convenience
from . import core
from . import hallucination
from . import visualization
from . import utils

__all__ = ['core', 'hallucination', 'visualization', 'utils']
