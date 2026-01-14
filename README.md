# TracTec

A Python package for computing lithospheric structure through geological time using plate tectonic reconstructions.

## Overview

TracTec provides two main capabilities:

1. **Seafloor Age Tracking**: Compute oceanic lithosphere ages using Lagrangian particle tracking from mid-ocean ridges
2. **Point Rotation**: Rotate user-provided points (e.g., continental lithospheric structure) through geological time

Both return `PointCloud` objects with Cartesian XYZ coordinates, designed for integration with [gadopt](https://github.com/g-adopt/g-adopt).

## Dependencies

TracTec requires **pygplates**, which must be installed separately:

```bash
conda install -c conda-forge pygplates
```

## Installation

```bash
pip install -e .
```

## Quick Start

### Seafloor Age Computation

```python
from tractec import SeafloorAgeTracker

# One-shot computation
cloud = SeafloorAgeTracker.compute_ages(
    target_age=100,
    starting_age=200,
    rotation_files=['rotations.rot'],
    topology_files=['topologies.gpmlz'],
    continental_polygons='continents.gpmlz'
)

xyz = cloud.xyz                    # (N, 3) Cartesian coordinates
ages = cloud.get_property('age')   # Material ages in Myr
```

### Continental Point Rotation

```python
from tractec import PointCloud, PointRotator, PolygonFilter

# Create point cloud and filter to continents
cloud = PointCloud.from_latlon(latlon_array)
polygon_filter = PolygonFilter(polygon_files='continents.gpmlz', rotation_files=['rotations.rot'])
continental = polygon_filter.filter_inside(cloud, at_age=0.0)

# Rotate to past geological age
rotator = PointRotator(rotation_files=['rotations.rot'], static_polygons='static_polygons.gpmlz')
continental = rotator.assign_plate_ids(continental, at_age=0.0)
rotated = rotator.rotate(continental, from_age=0.0, to_age=50.0)

xyz = rotated.xyz  # Use with gadopt
```

## Attribution

The seafloor age tracking algorithm is inspired by:

> Karlsen, K.S., Domeier, M., Gaina, C., Conrad, C.P. (2020). **A tracer-based algorithm for automatic generation of seafloor age grids from plate tectonic reconstructions.** *Computers & Geosciences*, 140, 104508. https://doi.org/10.1016/j.cageo.2020.104508

## License

MIT License
