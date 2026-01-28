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
import h5py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from gtrack import PointCloud, PointRotator, PolygonFilter
# -

# ## Data File Paths
#
# Run `make data` and `make osf-data` in the examples directory to download
# the plate model and lithospheric thickness data. For the plate model, 
# we are using the plate model by Matthews et al. that can be downloaded
# from https://data.gadopt.org/demos/Matthews_et_al_410_0.tar.gz, and 
# the global lithospheric thickness map produced by Hoggard et al. 2022, 
# which is based on the surface tomography model SL2013sv.

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

# Lithospheric thickness data from SL2013sv seismic model
lithosphere_file = Path("./lithospheric_thickness_maps/global/SL2013sv/SL2013sv.nc")
# -

# ## Load Lithospheric Thickness Data
#
# We load lithospheric thickness from the SL2013sv seismic tomography model.
# The data is stored in a NetCDF/HDF5 file with a 0.5-degree global grid.
# We use h5py to read the file for Firedrake compatibility.

# +
# Load lithospheric thickness data using h5py
with h5py.File(lithosphere_file, 'r') as f:
    lon_data = f['lon'][:]  # 0-360 degrees
    lat_data = f['lat'][:]  # -90 to 90 degrees
    thickness = f['z'][:]   # Lithospheric thickness in km

# Convert longitude from 0-360 to -180 to 180
lon_data = np.where(lon_data > 180, lon_data - 360, lon_data)

# Create meshgrid and flatten to point arrays
lon_grid, lat_grid = np.meshgrid(lon_data, lat_data)
latlon = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

# Create PointCloud from lat/lon
cloud = PointCloud.from_latlon(latlon)

# Add lithospheric thickness as a property (convert km to meters)
cloud.add_property('lithospheric_thickness', thickness.ravel() * 1e3)

print(f"Loaded {cloud.n_points} points from {lithosphere_file.name}")
print(f"Thickness range: {thickness.min():.1f} - {thickness.max():.1f} km")
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
# Plot the rotated continental points at the target geological age,
# coloured by lithospheric thickness.

# +
lonlat = rotated_cloud.lonlat
thickness_km = rotated_cloud.get_property('lithospheric_thickness') / 1e3

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Mollweide())

scatter = ax.scatter(
    lonlat[:, 0], lonlat[:, 1],  # lon, lat
    c=thickness_km,
    s=1,
    cmap='viridis',
    vmin=50, vmax=250,
    transform=ccrs.PlateCarree()
)

ax.coastlines(resolution='110m', linewidth=0.5, color='gray', alpha=0.5)
ax.set_global()

cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                    pad=0.05, aspect=40, shrink=0.8)
cbar.set_label('Lithospheric Thickness (km)')

ax.set_title(f'Continental Lithosphere Rotated to {int(target_age)} Ma')
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
thickness = rotated_cloud.get_property('lithospheric_thickness')

print(f"XYZ array shape: {xyz.shape}")
print(f"Thickness array shape: {thickness.shape}")
print(f"Thickness range: {thickness.min()/1e3:.1f} - {thickness.max()/1e3:.1f} km")
# -
