"""
tractec: High-performance seafloor age tracking and point rotation.

This package provides tools for:
- Computing seafloor ages from plate tectonic reconstructions using Lagrangian
  particle tracking with pygplates' C++ backend (GPlately-compatible)
- Rotating user-provided points through geological time
- Filtering points by polygon containment (e.g., continental regions)

The main entry point is SeafloorAgeTracker, which provides both a stepwise
interface and a one-shot compute_ages() method.

Logging
-------
tractec uses Python's logging module. Control verbosity via environment variable:

    export TRACTEC_LOGLEVEL=INFO    # Progress messages
    export TRACTEC_LOGLEVEL=DEBUG   # Detailed debug output
    export TRACTEC_LOGLEVEL=WARNING # Quiet (default)

Or programmatically:

>>> from tractec import enable_verbose, enable_debug
>>> enable_verbose()  # Show progress messages

Example
-------
>>> from tractec import SeafloorAgeTracker, TracerConfig
>>>
>>> # Simple one-shot computation
>>> cloud = SeafloorAgeTracker.compute_ages(
...     target_age=100,
...     starting_age=200,
...     rotation_files=['rotations.rot'],
...     topology_files=['topologies.gpmlz']
... )
>>> ages = cloud.get_property('age')
>>>
>>> # Or stepwise for intermediate states
>>> tracker = SeafloorAgeTracker(
...     rotation_files=['rotations.rot'],
...     topology_files=['topologies.gpmlz']
... )
>>> tracker.initialize(starting_age=200)
>>> for age in range(199, -1, -1):
...     cloud = tracker.step_to(age)
"""

__version__ = "0.2.0"

# Logging configuration (import first to configure before other modules)
from .logging import (
    configure_logging,
    set_log_level,
    enable_verbose,
    enable_debug,
    disable_logging,
    get_logger,
)

# Core seafloor age functionality
from .config import TracerConfig
from .hpc_integration import SeafloorAgeTracker

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

# Utility modules (for advanced users)
from .mesh import (
    create_icosahedral_mesh,
    create_icosahedral_mesh_latlon,
    create_icosahedral_mesh_xyz,
    mesh_point_count,
)
from .mor_seeds import (
    generate_mor_seeds,
    generate_mor_seeds_with_plate_ids,
    get_ridge_geometries,
)
from .initial_conditions import (
    compute_initial_ages,
    default_age_distance_law,
)
from .boundaries import (
    ContinentalPolygonCache,
    ResolvedTopologyCache,
    extract_ridge_geometries,
    extract_ridge_points_latlon,
)

__all__ = [
    # Main API
    "SeafloorAgeTracker",
    "TracerConfig",
    # Logging
    "configure_logging",
    "set_log_level",
    "enable_verbose",
    "enable_debug",
    "disable_logging",
    "get_logger",
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
    # Mesh generation (advanced)
    "create_icosahedral_mesh",
    "create_icosahedral_mesh_latlon",
    "create_icosahedral_mesh_xyz",
    "mesh_point_count",
    # MOR seeds (advanced)
    "generate_mor_seeds",
    "generate_mor_seeds_with_plate_ids",
    "get_ridge_geometries",
    # Initial conditions (advanced)
    "compute_initial_ages",
    "default_age_distance_law",
    # Caching (advanced)
    "ContinentalPolygonCache",
    "ResolvedTopologyCache",
    "extract_ridge_geometries",
    "extract_ridge_points_latlon",
]
