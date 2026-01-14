# Simple One-Shot Seafloor Age Computation
# =========================================
#
# This example demonstrates the simplest way to compute seafloor
# ages using TracTec. The compute_ages() class method provides a
# functional interface that handles all setup internally.
#
# Use this approach when you need ages at a single geological age
# and don't need incremental updates.

# +
from pathlib import Path
import numpy as np

from tractec import SeafloorAgeTracker
# -

# ## Data File Paths
#
# Adjust these paths to match your GPlates data installation.

# +
data_dir = Path("/Applications/GPlates_2.4.0/GeoData/FeatureCollections/")

rotation_files = [
    str(data_dir / "Rotations/Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot")
]

topology_files = [
    str(data_dir / "Topologies/Global_250-0Ma_Rotations_2022_Optimisation_v1.2_Topologies.gpmlz")
]

continental_polygons = str(
    data_dir / "ContinentalPolygons/Global_EarthByte_GPlates_PresentDay_ContinentalPolygons.gpmlz"
)
# -

# ## Compute Ages in One Call
#
# The compute_ages() class method creates a tracker, initializes
# tracers at ridges, and evolves them to the target age - all in
# a single call. This is the simplest way to get seafloor ages.

# +
# Compute seafloor ages at 5 Ma, starting from 10 Ma
cloud = SeafloorAgeTracker.compute_ages(
    target_age=5,               # Get ages at 5 Ma
    starting_age=10,            # Start from 10 Ma
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    verbose=True
)
# -

# ## Access Results
#
# The result is a PointCloud with xyz coordinates and an 'age'
# property containing the material age of each tracer.

# +
# Get coordinates and ages
xyz = cloud.xyz                    # (N, 3) Cartesian coordinates
ages = cloud.get_property('age')   # (N,) material ages in Myr

print(f"\nResults:")
print(f"  Number of tracers: {len(xyz)}")
print(f"  XYZ shape: {xyz.shape}")
print(f"  Age range: {ages.min():.1f} - {ages.max():.1f} Myr")
print(f"  Mean age: {ages.mean():.1f} Myr")

# The xyz array can be used directly with gadopt's cKDTree
# interpolation for applying seafloor ages to your mesh
# -

# ## Convert to Lat/Lon if Needed
#
# While Cartesian coordinates are used internally (for gadopt),
# you can easily convert to lat/lon for visualization.

# +
latlon = cloud.latlon  # (N, 2) array with [lat, lon] in degrees

print(f"\nGeographic coordinates:")
print(f"  Latitude range: {latlon[:, 0].min():.1f} to {latlon[:, 0].max():.1f}")
print(f"  Longitude range: {latlon[:, 1].min():.1f} to {latlon[:, 1].max():.1f}")
# -

# ## Summary
#
# For simple use cases where you need seafloor ages at a single
# geological age:
#
# 1. Call SeafloorAgeTracker.compute_ages() with your parameters
# 2. Access cloud.xyz for Cartesian coordinates
# 3. Access cloud.get_property('age') for material ages
# 4. Pass xyz directly to gadopt's interpolation routines
