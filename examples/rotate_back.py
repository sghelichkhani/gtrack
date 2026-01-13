import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pygplates
from pathlib import Path

data_dir = Path("/Applications/GPlates_2.4.0/GeoData/FeatureCollections/")

# Static polygons: Used to assign plate IDs to points
continental_polygons = (
    data_dir /
    "ContinentalPolygons/Global_EarthByte_GPlates_PresentDay_ContinentalPolygons.gpmlz"
)

static_polygons = (
    data_dir /
    "StaticPolygons/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons.gpmlz"
)

# Rotation model: Contains finite rotations for plate motion
rotation_model = data_dir / \
    "Rotations/Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot"

# Reconstruction time (millions of years ago)
reconstruction_age = 50.0  # Ma

lons = np.linspace(-180.0, 180.0, 360)
lats = np.linspace(-90.0, 90.0, 180)

# Create meshgrid
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Flatten to 1D arrays for easier processing
lon_flat = lon_grid.flatten()
lat_flat = lat_grid.flatten()

# Load continental polygons (for determining if points are in continents)
continental_polygons = pygplates.FeatureCollection(
    continental_polygons.as_posix()
)

static_polygons = pygplates.FeatureCollection(
    static_polygons.as_posix()
)

rotation_model = pygplates.RotationModel(rotation_model.as_posix())

continental_partitioner = pygplates.PlatePartitioner(
    continental_polygons,
    rotation_model,
    reconstruction_time=0.0,  # Partition at present-day (time=0)
    sort_partitioning_plates=pygplates.SortPartitioningPlates.by_partition_type_then_plate_area
)

#
static_partitioner = pygplates.PlatePartitioner(
    static_polygons,
    rotation_model,
    reconstruction_time=0.0,
    sort_partitioning_plates=pygplates.SortPartitioningPlates.by_partition_type_then_plate_area
)


# Create PointOnSphere objects for all points
points = np.asarray(
    [pygplates.PointOnSphere(lat, lon) for lat, lon in zip(lat_flat, lon_flat)]
)

# Seeing which points are in continents
flgs_in_continents = np.array(
    [True if continental_partitioner.partition_point(p) else False for p in points])
points_in_continents = points[flgs_in_continents]

plate_id_info = [
    static_partitioner.partition_point(p).get_feature()
    .get_reconstruction_plate_id()
    for p in points_in_continents
]

# Create point features from points_in_continents with plate IDs
point_features = []
for i, point in enumerate(points_in_continents):
    # Create a feature for this point
    feature = pygplates.Feature()
    feature.set_geometry(point)

    # Set the plate ID from plate_id_info
    plate_id = plate_id_info[i]
    feature.set_reconstruction_plate_id(plate_id)

    point_features.append(feature)

# Reconstruct the features to the past time
reconstructed_features = []
pygplates.reconstruct(
    point_features,
    rotation_model,
    reconstructed_features,
    reconstruction_age
)

# Extract reconstructed coordinates
reconstructed_coords = []
for rec_feature in reconstructed_features:
    rec_geom = rec_feature.get_reconstructed_geometry()
    if rec_geom:
        lat_rec, lon_rec = rec_geom.to_lat_lon()
        reconstructed_coords.append((lat_rec, lon_rec))

reconstructed_coords = np.array(reconstructed_coords)

# Plot reconstructed coordinates with Mollweide projection
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Mollweide())
ax.scatter(
    reconstructed_coords[:, 1],
    reconstructed_coords[:, 0],
    s=20,
    alpha=0.7,
    transform=ccrs.PlateCarree()
)
ax.coastlines()
ax.gridlines()
ax.set_global()
plt.title(f'Reconstructed Points at {reconstruction_age} Ma')
plt.tight_layout()
plt.show()
