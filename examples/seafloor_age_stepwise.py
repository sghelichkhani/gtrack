# Stepwise Seafloor Age Computation with gtrack
# ================================================
#
# This example demonstrates how to use gtrack's SeafloorAgeTracker
# for computing seafloor ages using Lagrangian particle tracking.
# The stepwise interface is ideal for simulations that need to
# access ages at multiple intermediate geological ages, let's say
# for a mantle convection simulation that is using sequential data-assimilation
# approach.
#
# In this example we will:
# 1. Initialise tracers at a starting age (300 Ma).
# 2. Step through geological time toward present, visualising at selected ages.
# 3. Demonstrate checkpointing for saving and restoring state.
#
# For this example, you need GPlates data files. Adjust the paths
# below to match your system.

# +
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from gtrack import SeafloorAgeTracker, TracerConfig
# -

# ## Data File Paths
#
# gtrack requires three types of GPlates data files:
# - Rotation files (.rot): Define how plates move through time
# - Topology files (.gpmlz): Define plate boundaries
# - Continental polygons (.gpmlz): Define continental regions
#
# These files are typically distributed with GPlates or with publications done
# using GPlates. For this example we use the plate model of Matthews et al. that can be downloaded
# from https://data.gadopt.org/demos/Matthews_et_al_410_0.tar.gz.
# Run `make data` in the examples directory to download and extract the data.


# +
data_dir = Path("./Matthews_et_al_410_0")

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
# TracerConfig controls the simulation parameters. The parameters are grouped
# into several categories:
#
# **Time stepping**: The `time_step` controls how frequently tracers are
# advected forward in time. Smaller values give more accurate trajectories
# but increase computation time.
#
# **Mesh initialisation**: At the starting age, tracers are placed on a
# Fibonacci sphere mesh that covers the globe uniformly. The `default_mesh_points`
# controls the resolution - more points give higher resolution but require
# more memory and computation.
#
# **Initial age calculation**: For the initial ocean tracers (those not
# born at ridges), ages are estimated based on distance to the nearest
# ridge divided by half the spreading rate. The `initial_ocean_mean_spreading_rate`
# sets this assumed spreading rate.
#
# **MOR seed generation**: New tracers are born at mid-ocean ridges (MORs)
# at each time step. The `ridge_sampling_degrees` controls how densely
# the ridge is sampled, and `spreading_offset_degrees` offsets new tracers
# slightly from the ridge axis (to avoid numerical issues at the boundary).
#
# **Collision detection**: Tracers are removed when they collide with
# subduction zones or continental margins. The `velocity_delta_threshold`
# detects sudden velocity changes (indicating plate boundary crossing),
# and `distance_threshold_per_myr` sets how close a tracer must be to a
# boundary to be considered for removal.

# +
config = TracerConfig(
    # Time stepping
    time_step=1.0,  # Myr per step

    # Mesh initialisation - number of points on Fibonacci sphere
    # 10000 points gives ~115 km spacing, 40000 gives ~57 km spacing
    default_mesh_points=10000,

    # Initial age calculation
    initial_ocean_mean_spreading_rate=75.0,  # mm/yr

    # MOR seed generation
    ridge_sampling_degrees=4.0,    # Ridge point spacing in degrees
    spreading_offset_degrees=0.01,  # Offset from ridge axis in degrees

    # Collision detection thresholds
    velocity_delta_threshold=7.0,      # km/Myr velocity change to trigger check
    distance_threshold_per_myr=10.0,   # km/Myr proximity threshold
)
# -

# ## Initialise the Tracker
#
# The SeafloorAgeTracker is designed for **stepwise** simulations where you
# need tracer positions at multiple intermediate geological ages. It maintains
# the tracer state internally, allowing you to repeatedly call `step_to()` to
# advance tracers forward in time (i.e., decreasing geological age toward 0 Ma).
#
# The tracker is initialised with the GPlates data files: rotation files define
# plate motions through time, topology files define plate boundaries, and
# continental polygons (optional) define regions where oceanic tracers are removed.
#
# After creating the tracker, we call `initialize()` with a `starting_age`.
# This creates an initial set of tracers on a Fibonacci sphere mesh covering the
# ocean basins. Each tracer's initial age is estimated from its distance to
# the nearest mid-ocean ridge, divided by half the `initial_ocean_mean_spreading_rate`
# from the configuration. This provides a reasonable first guess for the age
# structure of oceanic lithosphere at the starting time.

# +
tracker = SeafloorAgeTracker(
    rotation_files=rotation_files,
    topology_files=topology_files,
    continental_polygons=continental_polygons,
    config=config,
)

# Initialise tracers at the starting geological age
starting_age = 300  # Ma
n_tracers = tracker.initialize(starting_age=starting_age)
print(f"Initialised {n_tracers} tracers at {starting_age} Ma")
# -

# ## Stepwise Evolution
#
# The `step_to()` method evolves tracers to a target geological age. It handles
# all the intermediate steps internally (using the configured `time_step`) and
# returns a PointCloud with the tracer positions and ages at the target time.
#
# We'll step through selected ages to demonstrate the evolution:
# - 295 Ma (5 Myr after start)
# - 280 Ma (20 Myr after start)
# - 100 Ma (200 Myr after start)
# - 0 Ma (present day)

# +
# We define a helper function to plot the tracer field at each age.
# This keeps the code clean and avoids repetition as we step through
# multiple geological ages.


def plot_tracers(cloud, geological_age):
    """Plot tracer positions coloured by material age."""
    lonlat = cloud.lonlat
    ages = cloud.get_property('age')

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Mollweide())

    scatter = ax.scatter(
        lonlat[:, 0], lonlat[:, 1],
        c=ages,
        s=1,
        cmap='viridis_r',
        vmin=0, vmax=max(50, ages.max()),
        transform=ccrs.PlateCarree()
    )

    ax.coastlines(resolution='110m', linewidth=0.5)
    ax.set_global()

    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal',
                        pad=0.05, aspect=40, shrink=0.8)
    cbar.set_label('Seafloor Age (Myr)')

    ax.set_title(f'Seafloor Tracers at {geological_age} Ma')
    plt.show()

    # Print statistics
    stats_dict = {
        'count': len(ages),
        'min_age': ages.min(),
        'max_age': ages.max(),
        'mean_age': ages.mean()
    }
    print(f"  Tracers: {stats_dict['count']}")
    print(
        f"  Age range: {stats_dict['min_age']:.1f} - {stats_dict['max_age']:.1f} Myr")
    print(f"  Mean age: {stats_dict['mean_age']:.1f} Myr")
# -

# ### After 5 Myr (295 Ma)


# +
cloud = tracker.step_to(295)
plot_tracers(cloud, 295)
# -

# ### After 20 Myr (280 Ma)

# +
cloud = tracker.step_to(280)
plot_tracers(cloud, 280)
# -

# ### At 100 Ma (200 Myr of evolution)

# +
cloud = tracker.step_to(100)
plot_tracers(cloud, 100)
# -

# ### Present Day (0 Ma)

# +
cloud = tracker.step_to(0)
plot_tracers(cloud, 0)
# -
