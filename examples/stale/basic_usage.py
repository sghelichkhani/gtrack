"""
Basic usage example for TracTec package.

This script demonstrates how to use the SeafloorAgeModel API to compute
seafloor age grids at specific times.
"""

import numpy as np
from tractec import SeafloorAgeModel, TracerConfig

# NOTE: This example requires actual plate reconstruction files.
# Update these paths to point to your data files:
ROTATION_FILES = ["path/to/rotations.rot"]
TOPOLOGY_FILES = [
    "path/to/topologies_mesozoic.gpmlz",
    "path/to/topologies_paleozoic.gpmlz",
]
CONTINENTAL_POLYGONS = "path/to/continents.gpmlz"


def example_basic():
    """
    Basic example: Compute age grid at 100 Ma starting from 200 Ma.
    """
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)

    # Initialize model with default configuration
    model = SeafloorAgeModel(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS
    )

    # Compute age grid
    ages, lons, lats = model.compute_age_grid(
        time=100,       # Target time in Ma
        start_time=200, # Start from 200 Ma
        resolution=1.0   # 1 degree resolution
    )

    print(f"\nAge grid shape: {ages.shape}")
    print(f"Longitude range: {lons[0]:.1f} to {lons[-1]:.1f}")
    print(f"Latitude range: {lats[0]:.1f} to {lats[-1]:.1f}")
    print(f"Age range: {np.nanmin(ages):.1f} to {np.nanmax(ages):.1f} Ma")


def example_custom_config():
    """
    Example with custom configuration parameters.
    """
    print("\n" + "=" * 70)
    print("Example 2: Custom Configuration")
    print("=" * 70)

    # Create custom configuration
    config = TracerConfig(
        ridge_resolution=25e3,        # 25 km (finer than default 50 km)
        subduction_resolution=10e3,   # 10 km (finer than default 20 km)
        ridge_offset=500,             # 500 m from ridge
        time_step=0.5,                # 0.5 Myr timesteps
    )

    # Initialize model with custom config
    model = SeafloorAgeModel(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS,
        config=config
    )

    # Get raw tracer positions (not gridded)
    tracers = model.get_tracers_at_time(
        time=50,
        start_time=100
    )

    print(f"\nNumber of tracers: {len(tracers)}")
    print(f"Tracer age range: {tracers[:, 3].min():.1f} to {tracers[:, 3].max():.1f} Ma")


def example_with_preloading():
    """
    Example with boundary pre-loading for better performance.
    """
    print("\n" + "=" * 70)
    print("Example 3: Pre-loading Boundaries")
    print("=" * 70)

    model = SeafloorAgeModel(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS
    )

    # Pre-load boundaries for better performance
    print("Pre-loading boundaries for 0-200 Ma...")
    model.preload_boundaries(range(0, 201))

    # Now computing age grids will be faster
    for time in [0, 50, 100, 150, 200]:
        print(f"\nComputing age grid at {time} Ma...")
        ages, lons, lats = model.compute_age_grid(
            time=time,
            start_time=200,
            resolution=2.0  # Coarse grid for speed
        )
        print(f"  Grid shape: {ages.shape}")
        print(f"  Valid cells: {np.sum(~np.isnan(ages))}")


def example_initial_condition():
    """
    Example using initial condition from previous computation.
    """
    print("\n" + "=" * 70)
    print("Example 4: Using Initial Conditions")
    print("=" * 70)

    model = SeafloorAgeModel(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS
    )

    # Compute tracers at 200 Ma
    print("Computing tracers at 200 Ma...")
    tracers_200 = model.get_tracers_at_time(time=200, start_time=300)

    # Save for later use
    np.save("tracers_200Ma.npy", tracers_200)
    print(f"Saved {len(tracers_200)} tracers to tracers_200Ma.npy")

    # Later: Load and use as initial condition
    tracers_200_loaded = np.load("tracers_200Ma.npy")

    print("\nComputing age grid at 100 Ma using loaded initial condition...")
    ages, lons, lats = model.compute_age_grid(
        time=100,
        initial_tracers=tracers_200_loaded,
        resolution=1.0
    )

    print(f"Age grid computed successfully!")
    print(f"  Grid shape: {ages.shape}")


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  TracTec: Seafloor Age Grid Generation - Example Usage".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    print("NOTE: These examples require actual plate reconstruction files.")
    print("      Update the file paths at the top of this script to run.")
    print("\n")

    # Uncomment to run examples (requires actual data files):
    # example_basic()
    # example_custom_config()
    # example_with_preloading()
    # example_initial_condition()

    print("\nUpdate file paths in this script and uncomment examples to run.")
