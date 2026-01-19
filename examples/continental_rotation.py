# Continental Point Rotation Through Geological Time
# ===================================================
#
# This example demonstrates how to rotate user-provided points
# through geological time using tractec's Point Rotation API.
# This is useful for reconstructing continental lithospheric
# structure at past geological ages.
#
# The workflow is:
# 1. Create a PointCloud from lat/lon coordinates
# 2. Filter to keep only continental points
# 3. Assign plate IDs based on position
# 4. Rotate to target geological age
# 5. Export for visualization or further processing

# +
from pathlib import Path
import numpy as np

from tractec import (
    PointCloud,
    PointRotator,
    PolygonFilter,
    save_points_gpml,
    save_points_numpy,
)
# -

# ## Data File Paths
#
# We need rotation files and polygon files from GPlates.
# Adjust these paths to match your installation.

# +
data_dir = Path("/Applications/GPlates_2.4.0/GeoData/FeatureCollections/")

rotation_file = str(
    data_dir / "Rotations/Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot"
)

continental_polygons_file = str(
    data_dir / "ContinentalPolygons/Global_EarthByte_GPlates_PresentDay_ContinentalPolygons.gpmlz"
)

static_polygons_file = str(
    data_dir / "StaticPolygons/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons.gpmlz"
)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
# -

# ## Create Initial Point Grid
#
# We start with a regular lat/lon grid of points. In a real
# application, these would be locations where you have
# continental lithospheric data (e.g., from seismic tomography).

# +
# Create a 2-degree resolution grid
lons = np.linspace(-180.0, 180.0, 180)
lats = np.linspace(-90.0, 90.0, 90)
lon_grid, lat_grid = np.meshgrid(lons, lats)
latlon = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

# Create PointCloud from lat/lon
cloud = PointCloud.from_latlon(latlon)

# Add a sample property (e.g., lithospheric depth from present-day data)
# In reality, this would come from your input data
cloud.add_property('lithospheric_depth', np.ones(cloud.n_points) * 100e3)

print(f"Created grid with {cloud.n_points} points")
# -

# ## Filter to Continental Points
#
# PolygonFilter keeps only points inside specified polygons.
# Here we use it to select continental points and discard
# oceanic ones.

# +
polygon_filter = PolygonFilter(
    polygon_files=continental_polygons_file,
    rotation_files=[rotation_file]
)

# Keep only points inside continental polygons at present day
continental_cloud = polygon_filter.filter_inside(cloud, at_age=0.0)

print(f"Filtered to {continental_cloud.n_points} continental points")
print(f"Removed {cloud.n_points - continental_cloud.n_points} oceanic points")
# -

# ## Assign Plate IDs
#
# Points must have plate IDs before rotation. The PointRotator
# uses static polygons to determine which plate each point
# belongs to.

# +
rotator = PointRotator(
    rotation_files=[rotation_file],
    static_polygons=static_polygons_file
)

# Assign plate IDs at present day
# remove_undefined=True removes points that don't fall on any plate
continental_cloud = rotator.assign_plate_ids(
    continental_cloud,
    at_age=0.0,
    remove_undefined=True
)

# Check unique plate IDs
unique_plates = np.unique(continental_cloud.plate_ids)
print(f"\nAssigned plate IDs to {continental_cloud.n_points} points")
print(f"Found {len(unique_plates)} unique plates")
# -

# ## Rotate to Past Geological Age
#
# Now we rotate the continental points to their positions at
# a past geological age. The rotation uses batched operations
# for efficiency.

# +
target_age = 50.0  # 50 Ma

rotated_cloud = rotator.rotate(
    continental_cloud,
    from_age=0.0,
    to_age=target_age
)

print(f"\nRotated {rotated_cloud.n_points} points to {target_age} Ma")

# Verify properties are preserved
assert 'lithospheric_depth' in rotated_cloud.properties
print("Properties preserved after rotation")
# -

# ## Access Results for gadopt
#
# The rotated coordinates can be used directly with gadopt's
# spatial interpolation for applying continental structure
# to your mesh.

# +
# For gadopt integration:
xyz = rotated_cloud.xyz                              # (N, 3) Cartesian
depths = rotated_cloud.get_property('lithospheric_depth')  # (N,) property

print(f"\nResults for gadopt:")
print(f"  XYZ array shape: {xyz.shape}")
print(f"  Depth array shape: {depths.shape}")

# Get geographic coordinates for visualization
latlon_rotated = rotated_cloud.latlon
print(f"\nGeographic extent at {target_age} Ma:")
print(f"  Latitude: {latlon_rotated[:, 0].min():.1f} to {latlon_rotated[:, 0].max():.1f}")
print(f"  Longitude: {latlon_rotated[:, 1].min():.1f} to {latlon_rotated[:, 1].max():.1f}")
# -

# ## Save Results
#
# tractec supports multiple output formats:
# - NumPy (.npz) for Python workflows
# - GPML (.gpml) for viewing in GPlates

# +
# Save as NumPy for further processing
numpy_file = output_dir / f"continental_points_{int(target_age)}Ma.npz"
save_points_numpy(rotated_cloud, numpy_file)
print(f"\nSaved: {numpy_file}")

# Save as GPML for viewing in GPlates
gpml_file = output_dir / f"continental_points_{int(target_age)}Ma.gpml"
save_points_gpml(rotated_cloud, gpml_file)
print(f"Saved: {gpml_file}")
# -

# ## Summary
#
# The continental rotation workflow:
#
# 1. Create PointCloud from your lat/lon data
# 2. Filter to continental regions with PolygonFilter
# 3. Assign plate IDs with PointRotator.assign_plate_ids()
# 4. Rotate to target age with PointRotator.rotate()
# 5. Access xyz coordinates for gadopt interpolation
#
# Key points:
# - Properties (lithospheric_depth, etc.) are preserved during rotation
# - Coordinates are Cartesian internally, matching gadopt's format
# - Points without valid plate IDs are removed with warning
