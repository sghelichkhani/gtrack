# Simple One-Shot Seafloor Age Computation
# =========================================
#
# This example demonstrates the simplest way to compute seafloor
# ages using tractec. The compute_ages() class method provides a
# functional interface that handles all setup internally.
#
# Use this approach when you need ages at a single geological age
# and don't need incremental updates.

# +
from tractec import SeafloorAgeTracker, TracerConfig
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
# -

# ## Data File Paths
#
# Adjust these paths to match your GPlates data installation.

# +
data_dir = Path("../") / "data/Plate_model"

rotation_files = [
    data_dir / "Global_EB_250-0Ma_GK07_Matthews++.rot",
    data_dir / "Global_EB_410-250Ma_GK07_Matthews++.rot"
]

topology_files = [
    data_dir / "Mesozoic-Cenozoic_plate_boundaries_Matthews++.gpml",
    data_dir / "Paleozoic_plate_boundaries_Matthews++.gpml",
    data_dir / "TopologyBuildingBlocks_Matthews++.gpml",
]

# Continental polygons are optional - if not provided, tracers are not
# removed when entering continental regions
continental_polygons = data_dir / \
    "ContPolys/PresentDay_ContinentalPolygons_Matthews++.shp"
# -

# ## Configuration
#
# TracerConfig controls all tracer parameters. The new API uses
# pygplates' C++ backend for efficient collision detection and
# point advection, matching GPlately's approach.

# +
config = TracerConfig(
    # Time stepping
    time_step=1.0,  # Time step size (Myr)

    # Mesh initialization - icosahedral mesh refinement level
    # Level 5 = ~10,242 points, Level 6 = ~40,962 points
    default_refinement_levels=5,

    # Initial age calculation
    initial_ocean_mean_spreading_rate=75.0,  # mm/yr (GPlately default)

    # MOR seed generation
    ridge_sampling_degrees=0.5,    # Ridge tessellation (~50 km at equator)
    spreading_offset_degrees=0.01,  # Offset from ridge (~1 km)

    # Collision detection (C++ backend - GPlately compatible)
    velocity_delta_threshold=7.0,      # km/Myr (converted to 0.7 cm/yr)
    distance_threshold_per_myr=10.0,   # km/Myr
)
# -

# ## Compute Ages in One Call
#
# The compute_ages() class method creates a tracker, initializes
# tracers at ridges, and evolves them to the target age - all in
# a single call. This is the simplest way to get seafloor ages.

# +
# Compute seafloor ages at 170 Ma, starting from 200 Ma
cloud = SeafloorAgeTracker.compute_ages(
    target_age=170,
    starting_age=200,
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=config,
    verbose=True
)
# -
# ## Access Results
#
# The result is a PointCloud with xyz coordinates and an 'age'
# property containing the material age of each tracer.

# +
# Get coordinates and ages
# (N, 2) array with [lon, lat] in degrees
xyz = cloud.xyz
lonlat = cloud.lonlat  # (N, 2) array with [lon, lat] in degrees
tracer_ages = cloud.get_property('age')   # (N,) material ages in Myr
# -

# ## Plotting of the results
#
# Plot the tracers directly as a scatter plot on a Mollweide projection.

# +
# Extract lon/lat from the cloud
lons = lonlat[:, 0]
lats = lonlat[:, 1]

# Create figure with Mollweide projection
fig = plt.figure(num=1, figsize=(12, 6))
ax = plt.axes(projection=ccrs.Mollweide())

# Plot tracers as scatter points
scatter = ax.scatter(
    lons, lats,
    c=tracer_ages,
    s=1,  # Small point size
    cmap='viridis_r',  # Reversed: young=yellow, old=purple
    vmin=0, vmax=30,
    transform=ccrs.PlateCarree()
)

# Add coastlines and set global extent
ax.coastlines(resolution='110m', linewidth=0.5)
ax.set_global()

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                    pad=0.05, aspect=40, shrink=0.8)
cbar.set_label('Seafloor Age (Myr)', fontsize=12)

ax.set_title(f'Seafloor Tracers at 170 Ma')

fig.tight_layout()
fig.savefig('seafloor_age_simple.png', dpi=150)
# -
