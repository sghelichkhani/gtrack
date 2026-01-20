# Continental Point Rotation Through Geological Time
# ===================================================
#
# This example demonstrates how to rotate user-provided points
# through geological time using gtrack's Point Rotation API.
# This is useful for reconstructing continental lithospheric
# structure at past geological ages.
#
# **How it works:**
#
# Each point on Earth belongs to a tectonic plate. To rotate points
# back in time, we need to:
# 1. Determine which plate each point belongs to (using static polygons)
# 2. Apply the appropriate rotation for that plate at the target age
#
# The rotation files contain Euler pole rotations for each plate at
# each geological age. By assigning plate IDs to points and then
# applying the corresponding rotations, we can reconstruct where
# continental material was located in the past.

# +
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from gtrack import PointCloud, PointRotator, PolygonFilter
# -

# ## Data File Paths
#
# Run `make data` in the examples directory to download and extract the data.

# +
data_dir = Path("./Matthews_et_al_410_0")

rotation_files = [
    data_dir / "Global_EB_250-0Ma_GK07_Matthews++.rot",
    data_dir / "Global_EB_410-250Ma_GK07_Matthews++.rot"
]

continental_polygons_file = data_dir / \
    "ContPolys/PresentDay_ContinentalPolygons_Matthews++.shp"

static_polygons_file = data_dir / \
    "StaticPolys/PresentDay_StaticPlatePolygons_Matthews++.shp"
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
    rotation_files=rotation_files
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
    rotation_files=rotation_files,
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
# a past geological age. The `rotate()` method applies the appropriate
# Euler pole rotation for each point based on its plate ID.

# +
target_age = 100.0  # Ma

rotated_cloud = rotator.rotate(
    continental_cloud,
    from_age=0.0,
    to_age=target_age
)

print(f"Rotated {rotated_cloud.n_points} points to {target_age} Ma")
# -

# ## Visualise the Result
#
# Plot the rotated continental points at the target geological age.

# +
lonlat = rotated_cloud.lonlat

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Mollweide())

ax.scatter(
    lonlat[:, 1], lonlat[:, 0],  # lon, lat
    c='saddlebrown',
    s=1,
    transform=ccrs.PlateCarree()
)

ax.coastlines(resolution='110m', linewidth=0.5, color='gray', alpha=0.5)
ax.set_global()
ax.set_title(f'Continental Points Rotated to {int(target_age)} Ma')
plt.show()

print(f"Points: {rotated_cloud.n_points}")
print(f"Properties preserved: {list(rotated_cloud.properties.keys())}")
# -

# ## Using Results with gadopt
#
# The rotated PointCloud provides Cartesian coordinates that can be
# used directly with gadopt's spatial interpolation.

# +
xyz = rotated_cloud.xyz
depths = rotated_cloud.get_property('lithospheric_depth')

print(f"XYZ array shape: {xyz.shape}")
print(f"Depth array shape: {depths.shape}")
# -
