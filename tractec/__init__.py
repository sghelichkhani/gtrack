"""
TracTec: High-performance seafloor age grid generation.

This package provides tools for computing seafloor ages from plate tectonic
reconstructions using Lagrangian particle tracking.
"""

__version__ = "0.1.0"

from .model import SeafloorAgeModel
from .config import TracerConfig
from .hpc_integration import HPCSeafloorAgeTracker, MemoryEfficientSeafloorAgeTracker

__all__ = [
    "SeafloorAgeModel",
    "TracerConfig",
    "HPCSeafloorAgeTracker",
    "MemoryEfficientSeafloorAgeTracker",
]
