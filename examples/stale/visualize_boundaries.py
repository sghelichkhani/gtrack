"""
Visualize ridge and subduction zone locations - equivalent to get_ridge_and_subduction_locations.py

This script extracts and visualizes mid-ocean ridge and subduction zone locations
through geological time using the optimized tractec package.

Generates:
- Boundary location data files (.npy format)
- Visualization plots (PNG images for website)
- Animation frames showing boundary evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

from tractec import SeafloorAgeModel, TracerConfig
from tractec.boundaries import BoundaryCache


################################################################################
#                               User Options                                   #
################################################################################

# Folder path to plate tectonic reconstruction
folder_path = "/Users/krikarls/Google Drive/age-grid-project/data"

# Path to rotation files
rotation_files = [folder_path + "/Rotations/Matthews_etal_GPC_2016_410-0Ma_GK07.rot"]

# Path to plate boundary files
plate_boundary_files = [
    folder_path + "/DynamicPolygons/Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies.gpmlz",
    folder_path + "/DynamicPolygons/Matthews_etal_GPC_2016_Paleozoic_PlateTopologies.gpmlz",
    folder_path + "/DynamicPolygons/Matthews_etal_GPC_2016_TopologyBuildingBlocks.gpmlz"
]

# Time range
start_time = 0      # Myr
end_time = 400      # Myr
time_step = 10      # Save/plot every N Myr

# Plate boundary sampling resolution
delta_R = 50e3      # Resolution for mid-ocean ridges [meters]
delta_S = 20e3      # Resolution for subduction zones [meters]

# Output settings
output_dir = Path("boundary_data")
plot_dir = Path("boundary_plots")

# Create output directories
output_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)


################################################################################
#                         Visualization Functions                              #
################################################################################

def plot_boundaries(ridge_points, subduction_points, time, output_path):
    """
    Create a visualization of ridge and subduction zone locations.

    Parameters
    ----------
    ridge_points : np.ndarray
        Ridge locations as (N, 3) XYZ or (N, 2) lat/lon
    subduction_points : np.ndarray
        Subduction locations as (M, 3) XYZ or (M, 2) lat/lon
    time : float
        Reconstruction time
    output_path : Path
        Path to save the plot
    """
    from tractec.geometry import XYZ2LatLon

    fig, ax = plt.subplots(figsize=(14, 7))

    # Convert XYZ to lat/lon if needed
    if ridge_points.shape[1] == 3:
        ridge_lats, ridge_lons = XYZ2LatLon(ridge_points)
    else:
        ridge_lats, ridge_lons = ridge_points[:, 0], ridge_points[:, 1]

    if subduction_points.shape[1] == 3:
        subd_lats, subd_lons = XYZ2LatLon(subduction_points)
    else:
        subd_lats, subd_lons = subduction_points[:, 0], subduction_points[:, 1]

    # Plot boundaries
    ax.scatter(ridge_lons, ridge_lats, c='red', s=1, alpha=0.6,
               label=f'Mid-Ocean Ridges ({len(ridge_lons)} points)')
    ax.scatter(subd_lons, subd_lats, c='blue', s=1, alpha=0.6,
               label=f'Subduction Zones ({len(subd_lons)} points)')

    # Formatting
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title(f'Plate Boundaries at {time} Ma', fontsize=14, fontweight='bold')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Add statistics box
    stats_text = f'Time: {time} Ma\n'
    stats_text += f'Ridges: {len(ridge_lons):,} points\n'
    stats_text += f'Subduction: {len(subd_lons):,} points'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {output_path}")


def plot_boundary_density(ridge_points, subduction_points, time, output_path):
    """
    Create a 2D density plot showing concentration of plate boundaries.

    Parameters
    ----------
    ridge_points : np.ndarray
        Ridge locations
    subduction_points : np.ndarray
        Subduction locations
    time : float
        Reconstruction time
    output_path : Path
        Path to save the plot
    """
    from tractec.geometry import XYZ2LatLon

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Convert to lat/lon
    if ridge_points.shape[1] == 3:
        ridge_lats, ridge_lons = XYZ2LatLon(ridge_points)
    else:
        ridge_lats, ridge_lons = ridge_points[:, 0], ridge_points[:, 1]

    if subduction_points.shape[1] == 3:
        subd_lats, subd_lons = XYZ2LatLon(subduction_points)
    else:
        subd_lats, subd_lons = subduction_points[:, 0], subduction_points[:, 1]

    # Ridge density
    if len(ridge_lons) > 0:
        h1 = ax1.hist2d(ridge_lons, ridge_lats, bins=[72, 36],
                        range=[[-180, 180], [-90, 90]],
                        cmap='Reds', cmin=1)
        plt.colorbar(h1[3], ax=ax1, label='Ridge Point Count')
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.set_title(f'Mid-Ocean Ridge Density at {time} Ma', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Subduction density
    if len(subd_lons) > 0:
        h2 = ax2.hist2d(subd_lons, subd_lats, bins=[72, 36],
                        range=[[-180, 180], [-90, 90]],
                        cmap='Blues', cmin=1)
        plt.colorbar(h2[3], ax=ax2, label='Subduction Point Count')
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.set_title(f'Subduction Zone Density at {time} Ma', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved density plot: {output_path}")


def plot_boundary_length_evolution(times, ridge_lengths, subd_lengths, output_path):
    """
    Plot the total length of ridges and subduction zones over time.

    Parameters
    ----------
    times : array
        Time points
    ridge_lengths : array
        Total ridge lengths
    subd_lengths : array
        Total subduction lengths
    output_path : Path
        Output path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(times, np.array(ridge_lengths) / 1000, 'o-', color='red',
            linewidth=2, markersize=6, label='Mid-Ocean Ridges')
    ax.plot(times, np.array(subd_lengths) / 1000, 's-', color='blue',
            linewidth=2, markersize=6, label='Subduction Zones')

    ax.set_xlabel('Time (Ma)', fontsize=12)
    ax.set_ylabel('Total Length (km)', fontsize=12)
    ax.set_title('Plate Boundary Length Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved evolution plot: {output_path}")


def estimate_boundary_length(points):
    """
    Estimate total length of boundary from point cloud.

    Parameters
    ----------
    points : np.ndarray
        Boundary points as XYZ coordinates

    Returns
    -------
    float
        Estimated total length in km
    """
    if len(points) < 2:
        return 0.0

    # Simple estimate: sum of distances between consecutive points
    diffs = points[1:] - points[:-1]
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances) / 1000  # Convert to km


################################################################################
#                              Main Algorithm                                  #
################################################################################

def main():
    """Extract and visualize boundary locations through time."""

    print("\n" + "=" * 70)
    print("  TracTec: Boundary Location Extraction and Visualization")
    print("=" * 70)
    print(f"\nTime range: {start_time} Ma → {end_time} Ma")
    print(f"Time step: {time_step} Myr")
    print(f"Ridge resolution: {delta_R/1000:.1f} km")
    print(f"Subduction resolution: {delta_S/1000:.1f} km")

    # Create model
    print("\nInitializing model...")
    config = TracerConfig(
        ridge_resolution=delta_R,
        subduction_resolution=delta_S
    )

    model = SeafloorAgeModel(
        rotation_files=rotation_files,
        topology_files=plate_boundary_files,
        continental_polygons=folder_path + "/ContinentalPolygons/Matthews_etal_GPC_2016_ContinentalPolygons.gpmlz",
        config=config
    )

    # Create boundary cache
    print("Creating boundary cache...")
    cache = BoundaryCache(
        model.topology_features,
        model.rotation_model,
        ridge_resolution=delta_R,
        subduction_resolution=delta_S
    )

    # Arrays to track evolution
    times = []
    ridge_lengths = []
    subd_lengths = []

    # Process each timestep
    print("\n" + "=" * 70)
    print("Extracting boundaries...")
    print("=" * 70)

    time_points = range(start_time, end_time + 1, time_step)

    for i, time in enumerate(time_points):
        print(f"\n[{i+1}/{len(time_points)}] Processing time: {time} Ma")

        # Get boundary locations
        ridge_points = cache.get_ridges(time, as_xyz=True)
        subduction_points = cache.get_subduction(time, as_xyz=True)

        print(f"  Ridges: {len(ridge_points)} points")
        print(f"  Subduction: {len(subduction_points)} points")

        # Save to file
        ridge_path = output_dir / f"ridge_segments_{time}Ma.npy"
        subd_path = output_dir / f"subduction_segments_{time}Ma.npy"
        np.save(ridge_path, ridge_points)
        np.save(subd_path, subduction_points)
        print(f"  Saved: {ridge_path.name}")
        print(f"  Saved: {subd_path.name}")

        # Create visualization
        plot_path = plot_dir / f"boundaries_{time}Ma.png"
        plot_boundaries(ridge_points, subduction_points, time, plot_path)

        # Create density plot
        density_path = plot_dir / f"boundary_density_{time}Ma.png"
        plot_boundary_density(ridge_points, subduction_points, time, density_path)

        # Track lengths
        times.append(time)
        ridge_lengths.append(estimate_boundary_length(ridge_points))
        subd_lengths.append(estimate_boundary_length(subduction_points))

    # Create evolution plot
    print("\n" + "=" * 70)
    print("Creating summary plots...")
    print("=" * 70)

    evolution_path = plot_dir / "boundary_length_evolution.png"
    plot_boundary_length_evolution(times, ridge_lengths, subd_lengths, evolution_path)

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    print(f"\nRidge lengths:")
    print(f"  Mean: {np.mean(ridge_lengths):.0f} km")
    print(f"  Min: {np.min(ridge_lengths):.0f} km (at {times[np.argmin(ridge_lengths)]} Ma)")
    print(f"  Max: {np.max(ridge_lengths):.0f} km (at {times[np.argmax(ridge_lengths)]} Ma)")

    print(f"\nSubduction lengths:")
    print(f"  Mean: {np.mean(subd_lengths):.0f} km")
    print(f"  Min: {np.min(subd_lengths):.0f} km (at {times[np.argmin(subd_lengths)]} Ma)")
    print(f"  Max: {np.max(subd_lengths):.0f} km (at {times[np.argmax(subd_lengths)]} Ma)")

    # Cache info
    memory_info = cache.get_memory_usage()
    print(f"\nCache statistics:")
    print(f"  Memory used: {memory_info['total_mb']:.1f} MB")
    print(f"  Timesteps cached: {memory_info['num_timesteps_cached']}")

    # Final summary
    print("\n" + "=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print(f"\nData files saved to: {output_dir}")
    print(f"  Ridge files: {len(list(output_dir.glob('ridge_*.npy')))}")
    print(f"  Subduction files: {len(list(output_dir.glob('subduction_*.npy')))}")

    print(f"\nPlots saved to: {plot_dir}")
    print(f"  Boundary plots: {len(list(plot_dir.glob('boundaries_*.png')))}")
    print(f"  Density plots: {len(list(plot_dir.glob('boundary_density_*.png')))}")
    print(f"  Evolution plot: 1")


################################################################################
#                         Create Animation Frames                              #
################################################################################

def create_animation_guide():
    """
    Print instructions for creating an animation from the plots.
    """
    print("\n" + "=" * 70)
    print("Creating Animation")
    print("=" * 70)
    print("\nTo create an animation from the generated plots, you can use:")
    print("\n1. Using ImageMagick:")
    print("   convert -delay 20 -loop 0 boundary_plots/boundaries_*.png animation.gif")
    print("\n2. Using FFmpeg:")
    print("   ffmpeg -framerate 5 -pattern_type glob -i 'boundary_plots/boundaries_*.png' \\")
    print("          -c:v libx264 -pix_fmt yuv420p boundaries_evolution.mp4")
    print("\n3. Using Python (matplotlib animation):")
    print("   See the create_animation() function below")


def create_animation():
    """
    Create an MP4 animation of boundary evolution (requires ffmpeg).

    Note: This is a template - uncomment and modify as needed.
    """
    try:
        import matplotlib.animation as animation

        # Load all boundary data files
        ridge_files = sorted(output_dir.glob('ridge_*.npy'))
        subd_files = sorted(output_dir.glob('subduction_*.npy'))

        if len(ridge_files) == 0:
            print("No boundary data files found. Run main() first.")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))

        def animate(i):
            """Animation function."""
            from tractec.geometry import XYZ2LatLon

            ax.clear()

            # Load data
            ridge_points = np.load(ridge_files[i])
            subd_points = np.load(subd_files[i])

            # Extract time from filename
            time_str = ridge_files[i].stem.split('_')[-1].replace('Ma', '')
            time = int(time_str)

            # Convert and plot
            if len(ridge_points) > 0:
                ridge_lats, ridge_lons = XYZ2LatLon(ridge_points)
                ax.scatter(ridge_lons, ridge_lats, c='red', s=1, alpha=0.6, label='Ridges')

            if len(subd_points) > 0:
                subd_lats, subd_lons = XYZ2LatLon(subd_points)
                ax.scatter(subd_lons, subd_lats, c='blue', s=1, alpha=0.6, label='Subduction')

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_title(f'Plate Boundaries at {time} Ma', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(ridge_files),
                                      interval=200, repeat=True)

        # Save
        output_file = plot_dir / 'boundaries_evolution.mp4'
        anim.save(output_file, writer='ffmpeg', fps=5, dpi=100)
        print(f"\nAnimation saved: {output_file}")

    except ImportError:
        print("\nNote: Animation creation requires ffmpeg")
        print("Install with: conda install ffmpeg")


if __name__ == "__main__":
    # Check if data paths exist
    if not Path(folder_path).exists():
        print("\n" + "!" * 70)
        print("ERROR: Data folder not found!")
        print("!" * 70)
        print(f"\nPlease update 'folder_path' at the top of this script to point")
        print(f"to your plate reconstruction data files.")
        print(f"\nCurrent path: {folder_path}")
        exit(1)

    # Run main extraction and visualization
    main()

    # Show animation instructions
    create_animation_guide()

    # Optionally create animation (uncomment if you have ffmpeg installed)
    # create_animation()

    print("\n✅ All done!\n")
