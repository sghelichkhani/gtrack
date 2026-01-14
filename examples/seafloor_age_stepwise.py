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

from tractec import SeafloorAgeTracker, TracerConfig, PointCloudCheckpoint
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

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
# -

# ## Configuration
#
# TracerConfig controls the simulation parameters. The time_step
# determines how finely we resolve the evolution. Smaller time
# steps give more accurate results but take longer.

config = TracerConfig(
    time_step=1.0,              # 1 Myr time step
    ridge_resolution=50e3,      # 50 km ridge sampling
    subduction_resolution=20e3  # 20 km subduction zone sampling
)

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
    preload_boundaries=True,  # Pre-load for better performance
    verbose=True
)

# Initialize tracers at ridges for starting age
starting_age = 10  # 10 Ma - small for demo purposes
n_tracers = tracker.initialize(starting_age=starting_age)
print(f"Initialized {n_tracers} tracers at {starting_age} Ma")
# -

# ## Stepwise Evolution
#
# The step_to() method evolves tracers to a target geological age.
# It returns a PointCloud with an 'age' property containing the
# material age (time since formation at ridge) of each tracer.

# +
# Step through time in 2 Myr increments
for target_age in range(starting_age - 2, -1, -2):
    cloud = tracker.step_to(target_age)

    # Access data for gadopt integration
    xyz = cloud.xyz                    # (N, 3) Cartesian coordinates
    ages = cloud.get_property('age')   # (N,) material ages

    # Print statistics
    stats = tracker.get_statistics()
    print(f"\nAt {target_age} Ma:")
    print(f"  Tracers: {stats['count']}")
    print(f"  Age range: {stats['min_age']:.1f} - {stats['max_age']:.1f} Myr")
    print(f"  Mean age: {stats['mean_age']:.1f} Myr")
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
