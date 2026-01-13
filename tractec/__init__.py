"""
TracTec: High-performance seafloor age grid generation and point rotation.

This package provides tools for:
- Computing seafloor ages from plate tectonic reconstructions using Lagrangian
  particle tracking
- Rotating user-provided points through geological time
- Filtering points by polygon containment (e.g., continental regions)
"""

__version__ = "0.1.0"

# Core seafloor age functionality
from .model import SeafloorAgeModel
from .config import TracerConfig
from .hpc_integration import HPCSeafloorAgeTracker, MemoryEfficientSeafloorAgeTracker

# Point rotation API
from .point_rotation import PointCloud, PointRotator
from .polygon_filter import PolygonFilter
from .io_formats import (
    load_points_numpy,
    load_points_latlon,
    load_points_gpml,
    save_points_numpy,
    save_points_latlon,
    save_points_gpml,
    PointCloudCheckpoint,
)

__all__ = [
    # Seafloor age
    "SeafloorAgeModel",
    "TracerConfig",
    "HPCSeafloorAgeTracker",
    "MemoryEfficientSeafloorAgeTracker",
    # Point rotation
    "PointCloud",
    "PointRotator",
    "PolygonFilter",
    # IO utilities
    "load_points_numpy",
    "load_points_latlon",
    "load_points_gpml",
    "save_points_numpy",
    "save_points_latlon",
    "save_points_gpml",
    "PointCloudCheckpoint",
]
