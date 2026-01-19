# Stepwise Seafloor Age Computation with TracTec
# ================================================
#
# This example demonstrates how to use TracTec's SeafloorAgeTracker
# for computing seafloor ages using Lagrangian particle tracking.
# The stepwise interface is ideal for simulations that need to
# access ages at multiple intermediate geological ages.
#
# We will:
# 1. Initialize tracers at mid-ocean ridges for a starting age
# 2. Step through geological time toward present
# 3. Save and load checkpoints for restarts
# 4. Access results as PointCloud objects for gadopt integration
#
# For this example, you need GPlates data files. Adjust the paths
# below to match your system.

# +
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pygplates

from tractec import SeafloorAgeTracker, TracerConfig, PointCloudCheckpoint
from tractec.boundaries import extract_ridge_points_latlon, extract_subduction_points_latlon
# -

# ## Data File Paths
#
# TracTec requires three types of GPlates data files:
# - Rotation files (.rot): Define how plates move through time
# - Topology files (.gpmlz): Define plate boundaries
# - Continental polygons (.gpmlz): Define continental regions
#
# These files are typically distributed with GPlates. Adjust these
# paths to match your GPlates installation.

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

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
# -

# ## Configuration
#
# TracerConfig controls the simulation parameters. The new API uses
# pygplates' C++ backend for efficient point advection and collision
# detection, matching GPlately's approach.

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
    ridge_sampling_degrees=2.0,    # Ridge tessellation (~50 km at equator)
    spreading_offset_degrees=0.01,  # Offset from ridge (~1 km)

    # Collision detection (C++ backend - GPlately compatible)
    velocity_delta_threshold=7.0,      # km/Myr (converted to 0.7 cm/yr)
    distance_threshold_per_myr=10.0,   # km/Myr
)
# -

# ## Initialize the Tracker
#
# The SeafloorAgeTracker maintains tracer state in memory and
# provides incremental updates as geological age decreases
# toward present (0 Ma).

# +
tracker = SeafloorAgeTracker(
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=config,
    verbose=True
)

# Initialize tracers at ridges for starting age
starting_age = 200  # Ma
n_tracers = tracker.initialize(starting_age=starting_age)
print(f"Initialized {n_tracers} tracers at {starting_age} Ma")

# Load rotation model and topology features for boundary visualization
rotation_model = pygplates.RotationModel([str(f) for f in rotation_files])
topology_features = pygplates.FeatureCollection()
for f in topology_files:
    topology_features.add(pygplates.FeatureCollection(str(f)))
# -

# ## Stepwise Evolution
#
# The step_to() method evolves tracers to a target geological age.
# It returns a PointCloud with an 'age' property containing the
# material age (time since formation at ridge) of each tracer.

# +
# Step through time in 5 Myr increments
for target_age in range(starting_age - 5, -1, -5):
    # Get "before" tracer positions (current state before stepping)
    before_cloud = tracker.get_current_state()
    before_lonlat = before_cloud.lonlat
    before_ages = before_cloud.get_property('age')
    current_age = tracker.current_age

    # Get ridge and subduction points for visualization (BEFORE)
    ridge_lats, ridge_lons = extract_ridge_points_latlon(
        current_age, topology_features, rotation_model, tessellate_degrees=1.0
    )
    sub_lats, sub_lons = extract_subduction_points_latlon(
        current_age, topology_features, rotation_model, tessellate_degrees=1.0
    )

    # Step to target age
    cloud = tracker.step_to(target_age)

    # Access data for gadopt integration
    xyz = cloud.xyz                    # (N, 3) Cartesian coordinates
    ages = cloud.get_property('age')   # (N,) material ages
    lonlat = cloud.lonlat              # (N, 2) lon/lat coordinates

    # Get ridge and subduction points after step
    ridge_lats_after, ridge_lons_after = extract_ridge_points_latlon(
        target_age, topology_features, rotation_model, tessellate_degrees=1.0
    )
    sub_lats_after, sub_lons_after = extract_subduction_points_latlon(
        target_age, topology_features, rotation_model, tessellate_degrees=1.0
    )

    # Print statistics
    stats = tracker.get_statistics()
    print(f"\nAt {target_age} Ma:")
    print(f"  Tracers: {stats['count']}")
    print(f"  Age range: {stats['min_age']:.1f} - {stats['max_age']:.1f} Myr")
    print(f"  Mean age: {stats['mean_age']:.1f} Myr")

    # Plot tracers with boundaries - BEFORE and AFTER
    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             subplot_kw={'projection': ccrs.Mollweide()})

    # LEFT PLOT: Before step (at current_age)
    ax = axes[0]

    # Plot tracers BEFORE
    scatter = ax.scatter(
        before_lonlat[:, 0], before_lonlat[:, 1],
        c=before_ages,
        s=2,
        cmap='viridis_r',
        vmin=0, vmax=max(10, before_ages.max()) if len(before_ages) > 0 else 10,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # Plot ridges as black dots (scatter avoids dateline artifacts)
    if len(ridge_lons) > 0:
        ax.scatter(ridge_lons, ridge_lats, c='black', s=3, marker='.',
                   transform=ccrs.PlateCarree(), zorder=10, label='Ridges')

    # Plot subduction zones as triangles
    if len(sub_lons) > 0:
        ax.scatter(sub_lons, sub_lats, c='red', s=8, marker='^',
                   transform=ccrs.PlateCarree(), zorder=10, label='Subduction')

    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.set_global()
    ax.set_title(
        f'BEFORE: Tracers at {current_age} Ma\n(ridges=dots, subduction=triangles)')
    ax.legend(loc='lower left', fontsize=8)

    # RIGHT PLOT: After step (at target_age)
    ax = axes[1]

    # Plot tracers AFTER
    scatter2 = ax.scatter(
        lonlat[:, 0], lonlat[:, 1],
        c=ages,
        s=2,
        cmap='viridis_r',
        vmin=0, vmax=max(10, ages.max()) if len(ages) > 0 else 10,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # Plot ridges as black dots (at target_age)
    if len(ridge_lons_after) > 0:
        ax.scatter(ridge_lons_after, ridge_lats_after, c='black', s=3, marker='.',
                   transform=ccrs.PlateCarree(), zorder=10, label='Ridges')

    # Plot subduction zones as triangles (at target_age)
    if len(sub_lons_after) > 0:
        ax.scatter(sub_lons_after, sub_lats_after, c='red', s=8, marker='^',
                   transform=ccrs.PlateCarree(), zorder=10, label='Subduction')

    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.set_global()
    ax.set_title(
        f'AFTER: Tracers at {target_age} Ma\n(ridges=dots, subduction=triangles)')
    ax.legend(loc='lower left', fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(scatter2, ax=axes, orientation='horizontal',
                        pad=0.05, aspect=40, shrink=0.6)
    cbar.set_label('Seafloor Age (Myr)', fontsize=12)

    fig.suptitle(f'Step from {current_age} Ma to {target_age} Ma', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f'seafloor_age_{target_age:03d}Ma.png', dpi=150)
    plt.close(fig)
    print(f"  Saved plot: seafloor_age_{target_age:03d}Ma.png")


# -

# ## Checkpointing
#
# For long-running simulations, checkpointing allows you to save
# and restore state. This is useful for restarts after failures
# or for pausing/resuming simulations.

# +
# Save a checkpoint
checkpoint_file = output_dir / "seafloor_checkpoint.npz"
tracker.save_checkpoint(str(checkpoint_file))
print(f"\nSaved checkpoint at {tracker.current_age} Ma")

# To demonstrate restart, create a new tracker and load checkpoint
tracker2 = SeafloorAgeTracker(
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=config,
    verbose=False
)
tracker2.load_checkpoint(str(checkpoint_file))
print(f"Restored tracker at {tracker2.current_age} Ma with {tracker2.n_tracers} tracers")
# -

# ## Initialize from PointCloud
#
# For custom scenarios, you can initialize from an existing
# PointCloud. This is useful when you have tracer positions
# from another source.

# +
# Get current state as PointCloud
current_cloud = tracker.get_current_state()

# Create a new tracker and initialize from this cloud
tracker3 = SeafloorAgeTracker(
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=config,
    verbose=False
)

# Initialize from existing cloud
# The cloud must have 'age' property
tracker3.initialize_from_cloud(current_cloud, tracker.current_age)
print(f"\nInitialized from PointCloud: {tracker3.n_tracers} tracers at {tracker3.current_age} Ma")
# -

# ## Using Results in gadopt
#
# The PointCloud xyz coordinates are directly compatible with
# gadopt's spatial interpolation. Here's how you would typically
# use them:

# +
# Get final state
final_cloud = tracker.get_current_state()

# For gadopt integration:
xyz = final_cloud.xyz                  # Use directly with gadopt interpolation
ages = final_cloud.get_property('age')  # Material ages (time since formation)

# Convert ages to lithospheric depth (example transformation)
# In reality, you would use a proper age-to-depth relationship
lithospheric_depth = 10e3 * np.sqrt(ages)  # Simplified half-space cooling

print(f"\nFinal state at {tracker.current_age} Ma:")
print(f"  Points: {len(xyz)}")
print(f"  XYZ shape: {xyz.shape}")
print(f"  Age range: {ages.min():.1f} - {ages.max():.1f} Myr")
print(f"  Depth range: {lithospheric_depth.min()/1e3:.1f} - {lithospheric_depth.max()/1e3:.1f} km")
# -
