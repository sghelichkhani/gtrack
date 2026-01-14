"""
Example: Rotate continental points through geological time.

This example demonstrates the new Point Rotation API:
1. Create a grid of points
2. Filter to keep only continental points using PolygonFilter
3. Assign plate IDs using PointRotator
4. Rotate points to a past geological age
5. Export in multiple formats

This example uses the same data as rotate_back.py but with the cleaner API.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path

# Import the new tractec Point Rotation API
from tractec import (
    PointCloud,
    PointRotator,
    PolygonFilter,
    save_points_gpml,
    save_points_numpy,
    PointCloudCheckpoint,
)

# Path to GPlates data (adjust for your system)
data_dir = Path("/Applications/GPlates_2.4.0/GeoData/FeatureCollections/")

# File paths
continental_polygons_file = (
    data_dir /
    "ContinentalPolygons/Global_EarthByte_GPlates_PresentDay_ContinentalPolygons.gpmlz"
)
static_polygons_file = (
    data_dir /
    "StaticPolygons/Global_EarthByte_GPlates_PresentDay_StaticPlatePolygons.gpmlz"
)
rotation_file = (
    data_dir /
    "Rotations/Zahirovic_etal_2022_OptimisedMantleRef_and_NNRMantleRef.rot"
)

# Target geological age (Ma)
target_age = 50.0  # 50 million years ago


def main():
    print("=" * 60)
    print("Point Rotation API Example")
    print("=" * 60)

    # =========================================================================
    # Step 1: Create initial point cloud from a lat/lon grid
    # =========================================================================
    print("\n1. Creating initial point grid...")

    lons = np.linspace(-180.0, 180.0, 180)  # 2-degree resolution
    lats = np.linspace(-90.0, 90.0, 90)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    latlon = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # Create PointCloud from lat/lon
    cloud = PointCloud.from_latlon(latlon)
    print(f"   Created cloud with {cloud.n_points} points")

    # Add a dummy property (e.g., lithospheric depth placeholder)
    # In a real use case, this might be loaded from user data
    cloud.add_property('lithospheric_depth', np.ones(cloud.n_points) * 100e3)

    # =========================================================================
    # Step 2: Filter to continental points only
    # =========================================================================
    print("\n2. Filtering to continental points...")

    # Create polygon filter with continental polygons
    polygon_filter = PolygonFilter(
        polygon_files=str(continental_polygons_file),
        rotation_files=[str(rotation_file)]
    )

    # Keep only points inside continental polygons at present day (age=0)
    continental_cloud = polygon_filter.filter_inside(cloud, at_age=0.0)
    print(f"   Filtered to {continental_cloud.n_points} continental points")
    print(f"   (Removed {cloud.n_points - continental_cloud.n_points} oceanic points)")

    # =========================================================================
    # Step 3: Assign plate IDs
    # =========================================================================
    print("\n3. Assigning plate IDs...")

    # Create rotator with static polygons for plate ID assignment
    rotator = PointRotator(
        rotation_files=[str(rotation_file)],
        static_polygons=str(static_polygons_file)
    )

    # Assign plate IDs at present day
    # remove_undefined=True will remove points that don't fall on any plate
    continental_cloud = rotator.assign_plate_ids(
        continental_cloud,
        at_age=0.0,
        remove_undefined=True
    )
    print(f"   Assigned plate IDs to {continental_cloud.n_points} points")

    # Show unique plate IDs
    unique_plates = np.unique(continental_cloud.plate_ids)
    print(f"   Found {len(unique_plates)} unique plate IDs")

    # =========================================================================
    # Step 4: Rotate to target age
    # =========================================================================
    print(f"\n4. Rotating to {target_age} Ma...")

    # Rotate from present day (age=0) to target age
    rotated_cloud = rotator.rotate(
        continental_cloud,
        from_age=0.0,
        to_age=target_age
    )
    print(f"   Rotated {rotated_cloud.n_points} points to {target_age} Ma")

    # Verify properties are preserved
    assert 'lithospheric_depth' in rotated_cloud.properties
    print("   Properties preserved after rotation")

    # =========================================================================
    # Step 5: Save outputs
    # =========================================================================
    print("\n5. Saving outputs...")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save as GPML for viewing in GPlates
    gpml_file = output_dir / f"continental_points_{target_age}Ma.gpml"
    save_points_gpml(rotated_cloud, gpml_file)
    print(f"   Saved GPML: {gpml_file}")

    # Save as numpy for further processing
    numpy_file = output_dir / f"continental_points_{target_age}Ma.npz"
    save_points_numpy(rotated_cloud, numpy_file)
    print(f"   Saved numpy: {numpy_file}")

    # Save checkpoint with metadata
    checkpoint = PointCloudCheckpoint()
    checkpoint_file = output_dir / f"checkpoint_{target_age}Ma.npz"
    checkpoint.save(
        rotated_cloud,
        checkpoint_file,
        geological_age=target_age,
        metadata={
            'source_age': 0.0,
            'n_original_points': cloud.n_points,
            'filter': 'continental_only'
        }
    )
    print(f"   Saved checkpoint: {checkpoint_file}")

    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    print("\n6. Creating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             subplot_kw={'projection': ccrs.Mollweide()})

    # Present day positions
    ax1 = axes[0]
    latlon_present = continental_cloud.latlon
    ax1.scatter(
        latlon_present[:, 1],  # lon
        latlon_present[:, 0],  # lat
        s=5, alpha=0.5, c='blue',
        transform=ccrs.PlateCarree()
    )
    ax1.coastlines()
    ax1.gridlines()
    ax1.set_global()
    ax1.set_title('Present Day (0 Ma)')

    # Rotated positions
    ax2 = axes[1]
    latlon_rotated = rotated_cloud.latlon
    ax2.scatter(
        latlon_rotated[:, 1],  # lon
        latlon_rotated[:, 0],  # lat
        s=5, alpha=0.5, c='red',
        transform=ccrs.PlateCarree()
    )
    ax2.coastlines()
    ax2.gridlines()
    ax2.set_global()
    ax2.set_title(f'Reconstructed at {target_age} Ma')

    plt.tight_layout()

    plot_dir = Path("plots")
    plot_dir.mkdir(exist_ok=True)
    plot_file = plot_dir / f"continental_rotation_{target_age}Ma.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"   Saved plot: {plot_file}")
    plt.close()

    # =========================================================================
    # Step 7: Demonstrate incremental rotation
    # =========================================================================
    print("\n7. Demonstrating incremental rotation...")

    # For very long time spans, use incremental rotation
    # This handles plate ID changes as points cross plate boundaries
    if target_age > 50:
        incremental_result = rotator.rotate_incremental(
            continental_cloud,
            from_age=0.0,
            to_age=target_age,
            time_step=5.0,  # 5 Myr steps
            reassign_at_each_step=True
        )
        print(f"   Incremental rotation completed: {incremental_result.n_points} points")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  - Started with {cloud.n_points} points")
    print(f"  - Filtered to {continental_cloud.n_points} continental points")
    print(f"  - Rotated to {target_age} Ma")
    print(f"  - XYZ array shape: {rotated_cloud.xyz.shape}")
    print(f"  - For gadopt: use rotated_cloud.xyz directly")
    print("=" * 60)

    # For gadopt integration, you can directly use:
    xyz_for_gadopt = rotated_cloud.xyz  # (N, 3) Cartesian array
    print(f"\nXYZ array for gadopt: shape={xyz_for_gadopt.shape}")


if __name__ == "__main__":
    main()
