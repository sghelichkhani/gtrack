"""
Generate seafloor age grids - equivalent to generate_seafloor.py

This script demonstrates the full workflow of generating seafloor age grids
through geological time, now using the optimized tractec package API.

Generates:
- Age grid data files (.npy or .xyz format)
- Visualization plots (PNG images for website)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tractec import SeafloorAgeModel, TracerConfig


################################################################################
#                               User Options                                   #
################################################################################

# Folder path to plate tectonic reconstruction
folder_path = Path( "./../data/mathews2016/")

# Set path to rotation files
rotation_files = [folder_path / "Matthews_etal_GPC_2016_410-0Ma_GK07.rot"]

# Set path to plate boundary files
plate_boundary_files = [
    folder_path / "Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies.gpmlz",
    folder_path / "Matthews_etal_GPC_2016_Paleozoic_PlateTopologies.gpmlz",
    # folder_path / "Matthews_etal_GPC_2016_TopologyBuildingBlocks.gpmlz"
]

# Set path to continental polygon files
continent_polygons = "/Users/sghelichkhani/Workplace/pygplates-tutorials/data/workshop/ContinentalPolygons/Seton_etal_ESR2012_ContinentalPolygons_2012.1.gpmlz" # folder_path / "Matthews_etal_GPC_2016_ContinentalPolygons.gpmlz"


# Time-stepping parameters
start_time = 400  # [Myr] Must be an integer larger than end_time
end_time = 0      # [Myr] Must be an integer smaller than start_time

# Parameters for tracer computation
delta_R = 50e3      # Ridge resolution [meters]
epsilon_R = 1e3     # Ridge-offset distance for tracers [meters]
delta_S = 20e3      # Subduction zone resolution [meters]

# Output settings
save_freq = 5       # Save results every [save_freq] Myrs
output_dir = Path("output")
plot_dir = Path("plots")
save_format = "xyz"  # "xyz" or "npy"

# Create output directories
output_dir.mkdir(exist_ok=True)
plot_dir.mkdir(exist_ok=True)


################################################################################
#                         Visualization Functions                              #
################################################################################

def plot_age_grid(ages, lons, lats, time, output_path):
    """
    Create a visualization of the seafloor age grid.

    Parameters
    ----------
    ages : np.ndarray
        2D array of ages
    lons : np.ndarray
        Longitude coordinates
    lats : np.ndarray
        Latitude coordinates
    time : float
        Reconstruction time
    output_path : Path
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create meshgrid for plotting
    LON, LAT = np.meshgrid(lons, lats)

    # Plot age grid
    im = ax.pcolormesh(LON, LAT, ages, cmap='viridis_r', shading='auto',
                       vmin=0, vmax=200)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Seafloor Age (Ma)')

    # Formatting
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_title(f'Seafloor Age Grid at {time} Ma', fontsize=14, fontweight='bold')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.3)

    # Add statistics
    valid_ages = ages[~np.isnan(ages)]
    if len(valid_ages) > 0:
        stats_text = f'Mean: {np.mean(valid_ages):.1f} Ma\n'
        stats_text += f'Max: {np.max(valid_ages):.1f} Ma\n'
        stats_text += f'Coverage: {len(valid_ages) / ages.size * 100:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {output_path}")


def save_tracers(tracers, time, output_dir, save_format="xyz"):
    """
    Save tracers to file in specified format.

    Parameters
    ----------
    tracers : np.ndarray
        Tracer array (N, 4) with [x, y, z, age]
    time : float
        Reconstruction time
    output_dir : Path
        Output directory
    save_format : str
        Format: "xyz" or "npy"
    """
    from tractec.geometry import XYZ2LatLon

    lats, lons = XYZ2LatLon(tracers[:, :3])
    age = tracers[:, 3]
    data = np.column_stack([lons, lats, age])  # GMT default: lons-lats

    if save_format == "xyz":
        path = output_dir / f"tracers_{time}Ma.xyz"
        np.savetxt(path, data, fmt='%.6f', header='lon lat age(Ma)')
    else:
        path = output_dir / f"tracers_{time}Ma.npy"
        np.save(path, data)

    print(f"  Saved tracers: {path}")


################################################################################
#                              Main Algorithm                                  #
################################################################################

def main():
    """Run the seafloor age grid generation workflow."""

    print("\n" + "=" * 70)
    print("  TracTec: Seafloor Age Grid Generation")
    print("=" * 70)
    print(f"\nTime range: {start_time} Ma → {end_time} Ma")
    print(f"Save frequency: every {save_freq} Myr")
    print(f"Output directory: {output_dir}")
    print(f"Plot directory: {plot_dir}")

    # Create configuration
    config = TracerConfig(
        ridge_resolution=delta_R,
        subduction_resolution=delta_S,
        ridge_offset=epsilon_R,
        time_step=1.0
    )

    # Initialize model
    print("\nInitializing SeafloorAgeModel...")
    model = SeafloorAgeModel(
        rotation_files=rotation_files,
        topology_files=plate_boundary_files,
        continental_polygons=continent_polygons,
        config=config
    )

    # Pre-load boundaries for better performance
    # print(f"Pre-loading boundaries for {start_time}-{end_time} Ma...")
    model.preload_boundaries(range(end_time, start_time + 1))

    # # Get memory usage
    # memory_info = model._boundary_cache.get_memory_usage()
    # print(f"  Boundary cache: {memory_info['total_mb']:.1f} MB in memory")
    # print(f"  Cached timesteps: {memory_info['num_timesteps_cached']}")

    # # Initialize tracers at start time
    # print(f"\nInitializing tracers at {start_time} Ma...")
    tracers = model._initialize_tracers(start_time)
    print(f"  Initial tracer count: {len(tracers)}")

    current_time = start_time

    while current_time > end_time:
        next_time = current_time - 1

        print(f"\n--- Processing: {current_time} Ma → {next_time} Ma ---")
        print(f"Tracer count: {len(tracers)}")

        # Evolve tracers by one timestep
        tracers = model._evolve_tracers(tracers, current_time, next_time)

        print(current_time)
        # # Save results at specified frequency
        # if next_time % save_freq == 0:
        #     print(f"\nSaving results for {next_time} Ma...")
        #     save_tracers(tracers, next_time, output_dir, save_format)

        #     # Generate age grid and plot
        #     ages, lons, lats = model._tracers_to_grid(tracers, next_time, resolution=1.0)
        #     plot_path = plot_dir / f"age_grid_{next_time}Ma.png"
        #     # plot_age_grid(ages, lons, lats, next_time, plot_path)

        current_time = next_time



################################################################################
#                         Advanced Visualization                               #
################################################################################

def generate_summary_plots():
    """
    Generate summary visualizations after main computation.

    Creates:
    - Age evolution plot (multiple timesteps)
    - Age statistics over time
    """
    print("\n" + "=" * 70)
    print("Generating Summary Plots...")
    print("=" * 70)

    # Load all saved data files
    data_files = sorted(output_dir.glob(f'tracers_*.{save_format}'))

    if len(data_files) < 2:
        print("Not enough data files for summary plots")
        return

    # Extract times and statistics
    times = []
    mean_ages = []
    max_ages = []
    tracer_counts = []

    for data_file in data_files:
        # Extract time from filename
        time_str = data_file.stem.split('_')[1].replace('Ma', '')
        time = int(time_str)
        times.append(time)

        # Load data
        if save_format == "xyz":
            data = np.loadtxt(data_file)
        else:
            data = np.load(data_file)

        ages = data[:, 2]
        mean_ages.append(np.mean(ages))
        max_ages.append(np.max(ages))
        tracer_counts.append(len(ages))

    # Sort by time
    sorted_indices = np.argsort(times)
    times = np.array(times)[sorted_indices]
    mean_ages = np.array(mean_ages)[sorted_indices]
    max_ages = np.array(max_ages)[sorted_indices]
    tracer_counts = np.array(tracer_counts)[sorted_indices]

    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Age statistics
    ax1.plot(times, mean_ages, 'o-', label='Mean Age', linewidth=2)
    ax1.plot(times, max_ages, 's-', label='Max Age', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Reconstruction Time (Ma)', fontsize=12)
    ax1.set_ylabel('Seafloor Age (Ma)', fontsize=12)
    ax1.set_title('Seafloor Age Statistics Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tracer count
    ax2.plot(times, tracer_counts / 1000, 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Reconstruction Time (Ma)', fontsize=12)
    ax2.set_ylabel('Tracer Count (thousands)', fontsize=12)
    ax2.set_title('Active Tracer Count Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    summary_path = plot_dir / 'summary_statistics.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved summary plot: {summary_path}")


if __name__ == "__main__":
    # Check if data paths exist
    from pathlib import Path

    if not Path(folder_path).exists():
        print("\n" + "!" * 70)
        print("ERROR: Data folder not found!")
        print("!" * 70)
        print(f"\nPlease update 'folder_path' at the top of this script to point")
        print(f"to your plate reconstruction data files.")
        print(f"\nCurrent path: {folder_path}")
        print("\nThis script requires:")
        print("  - Rotation files (.rot)")
        print("  - Topology files (.gpmlz)")
        print("  - Continental polygons (.gpmlz)")
        exit(1)

    # Run main computation
    main()

    # Generate summary plots
    try:
        generate_summary_plots()
    except Exception as e:
        print(f"\nWarning: Could not generate summary plots: {e}")

    print("\n✅ All done!\n")
