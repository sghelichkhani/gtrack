# TracTec

A Python package for computing lithospheric structure through geological time using plate tectonic reconstructions.

## Overview

TracTec provides two main capabilities:

1. **Seafloor Age Tracking**: Compute oceanic lithosphere ages using Lagrangian particle tracking from mid-ocean ridges
2. **Point Rotation**: Rotate user-provided points (e.g., continental lithospheric structure) through geological time

Both return `PointCloud` objects with Cartesian XYZ coordinates, designed for integration with [gadopt](https://github.com/g-adopt/g-adopt).

## Based on GPlately

TracTec's seafloor age tracking algorithm is based on [GPlately](https://github.com/GPlates/gplately)'s `SeafloorGrid` implementation. The algorithm matches GPlately's approach for:

- Initial ocean mesh generation using icosahedral tessellation
- Mid-ocean ridge seed point generation using stage rotation poles
- Velocity-based collision detection at subduction zones
- Continental collision via polygon queries

## Performance

TracTec is written with performance in mind. All computationally intensive operations are delegated to **pygplates' C++ backend**:

- **Point reconstruction**: Uses `pygplates.TopologicalModel.reconstruct_geometry()` for efficient batch advection
- **Collision detection**: Uses pygplates' built-in `DefaultDeactivatePoints` with velocity/distance thresholds
- **Point-in-polygon queries**: Uses quad tree spatial indexing for efficient bulk queries
- **Batch point creation**: Uses `pygplates.MultiPointOnSphere` to minimize Python-C++ boundary crossings

## Installation

```bash
pip install tractec
```

For development:

```bash
pip install -e .
```

## Quick Start

### Example 1: One-Shot Seafloor Age Computation

The simplest way to compute seafloor ages at a single geological age:

```python
from tractec import SeafloorAgeTracker, TracerConfig

# Configure tracer parameters
config = TracerConfig(
    time_step=1.0,                    # Myr
    default_refinement_levels=5,      # ~10,000 mesh points
    ridge_sampling_degrees=0.5,       # ~50 km ridge tessellation
)

# Compute seafloor ages at 100 Ma, starting from 200 Ma
cloud = SeafloorAgeTracker.compute_ages(
    target_age=100,
    starting_age=200,
    rotation_files=['rotations.rot'],
    topology_files=['topologies.gpmlz'],
    continental_polygons='continents.gpmlz',  # optional
    config=config,
)

# Access results
xyz = cloud.xyz                    # (N, 3) Cartesian coordinates
ages = cloud.get_property('age')   # (N,) material ages in Myr
lonlat = cloud.lonlat              # (N, 2) lon/lat in degrees
```

### Example 2: Stepwise Seafloor Age Computation

For simulations that need ages at multiple intermediate geological ages:

```python
from tractec import SeafloorAgeTracker, TracerConfig

config = TracerConfig(time_step=1.0)

tracker = SeafloorAgeTracker(
    rotation_files=['rotations.rot'],
    topology_files=['topologies.gpmlz'],
    continental_polygons='continents.gpmlz',
    config=config,
)

# Initialize tracers at 200 Ma
tracker.initialize(starting_age=200)

# Step through time toward present
for target_age in range(195, -1, -5):
    cloud = tracker.step_to(target_age)

    # Access data at each timestep
    xyz = cloud.xyz
    ages = cloud.get_property('age')

    # Use with gadopt...

# Checkpointing for long runs
tracker.save_checkpoint('checkpoint.npz')
tracker.load_checkpoint('checkpoint.npz')
```

### Example 3: Rotating Continental Data Back in Time

Rotate present-day continental lithospheric structure to past geological ages:

```python
from tractec import PointCloud, PointRotator, PolygonFilter
import numpy as np

# Create point cloud from your data (e.g., seismic tomography grid)
latlon = np.array([[lat1, lon1], [lat2, lon2], ...])
cloud = PointCloud.from_latlon(latlon)
cloud.add_property('lithospheric_depth', depth_values)

# Filter to keep only continental points
polygon_filter = PolygonFilter(
    polygon_files='continental_polygons.gpmlz',
    rotation_files=['rotations.rot']
)
continental = polygon_filter.filter_inside(cloud, at_age=0.0)

# Assign plate IDs and rotate to 50 Ma
rotator = PointRotator(
    rotation_files=['rotations.rot'],
    static_polygons='static_polygons.gpmlz'
)
continental = rotator.assign_plate_ids(continental, at_age=0.0)
rotated = rotator.rotate(continental, from_age=0.0, to_age=50.0)

# Use rotated coordinates with gadopt
xyz = rotated.xyz
depths = rotated.get_property('lithospheric_depth')
```

## Examples

See the `examples/` directory for complete working examples:

- `seafloor_age_simple.py` - One-shot seafloor age computation
- `seafloor_age_stepwise.py` - Stepwise computation with visualization
- `continental_rotation.py` - Rotating continental structure back in time

## Attribution

The seafloor age tracking algorithm is based on GPlately and inspired by:

> Karlsen, K.S., Domeier, M., Gaina, C., Conrad, C.P. (2020). **A tracer-based algorithm for automatic generation of seafloor age grids from plate tectonic reconstructions.** *Computers & Geosciences*, 140, 104508. https://doi.org/10.1016/j.cageo.2020.104508

## License

MIT License
