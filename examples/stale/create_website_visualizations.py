"""
Create publication-quality visualizations for website display.

This script generates polished, web-ready visualizations of seafloor ages
and plate boundaries optimized for use on websites and presentations.

Generates:
- High-resolution age grid plots
- Multi-panel comparison plots
- Interactive HTML plots (optional with plotly)
- Time-lapse sequences
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path

from tractec import SeafloorAgeModel, TracerConfig


################################################################################
#                               Configuration                                  #
################################################################################

# Data paths - UPDATE THESE
folder_path = "/Users/krikarls/Google Drive/age-grid-project/data"

rotation_files = [folder_path + "/Rotations/Matthews_etal_GPC_2016_410-0Ma_GK07.rot"]
plate_boundary_files = [
    folder_path + "/DynamicPolygons/Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies.gpmlz",
    folder_path + "/DynamicPolygons/Matthews_etal_GPC_2016_Paleozoic_PlateTopologies.gpmlz",
    folder_path + "/DynamicPolygons/Matthews_etal_GPC_2016_TopologyBuildingBlocks.gpmlz"
]
continent_polygons = folder_path + "/ContinentalPolygons/Matthews_etal_GPC_2016_ContinentalPolygons.gpmlz"

# Output settings
output_dir = Path("website_plots")
output_dir.mkdir(exist_ok=True)

# Visualization settings
DPI = 300  # High resolution for website
FIGSIZE_SINGLE = (14, 7)
FIGSIZE_MULTI = (16, 10)


################################################################################
#                         Publication-Quality Plots                            #
################################################################################

def create_high_quality_age_grid(model, time, start_time, output_path):
    """
    Create a publication-quality age grid visualization.

    Parameters
    ----------
    model : SeafloorAgeModel
        The model instance
    time : float
        Target time
    start_time : float
        Starting time for evolution
    output_path : Path
        Output file path
    """
    print(f"\nGenerating high-quality age grid for {time} Ma...")

    # Compute age grid
    ages, lons, lats = model.compute_age_grid(
        time=time,
        start_time=start_time,
        resolution=0.5  # High resolution
    )

    # Create figure with custom styling
    fig = plt.figure(figsize=FIGSIZE_SINGLE)
    ax = fig.add_subplot(111)

    # Create meshgrid
    LON, LAT = np.meshgrid(lons, lats)

    # Plot with custom colormap
    im = ax.pcolormesh(LON, LAT, ages, cmap='RdYlBu_r', shading='auto',
                       vmin=0, vmax=200)

    # Add colorbar with custom styling
    cbar = plt.colorbar(im, ax=ax, label='Seafloor Age (Ma)', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Styling
    ax.set_xlabel('Longitude (°)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=13, fontweight='bold')
    ax.set_title(f'Seafloor Age Distribution at {time} Ma',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    # Add grid with custom styling
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Add statistics panel
    valid_ages = ages[~np.isnan(ages)]
    if len(valid_ages) > 0:
        stats_text = (f'Statistics:\n'
                     f'Mean age: {np.mean(valid_ages):.1f} Ma\n'
                     f'Median age: {np.median(valid_ages):.1f} Ma\n'
                     f'Max age: {np.max(valid_ages):.1f} Ma\n'
                     f'Coverage: {len(valid_ages) / ages.size * 100:.1f}%')

        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=props, family='monospace')

    # Add branding/attribution
    attribution = '© TracTec - Plate Reconstruction Analysis'
    ax.text(0.98, 0.02, attribution, transform=ax.transAxes,
            fontsize=8, verticalalignment='bottom', horizontalalignment='right',
            alpha=0.7, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")
    return ages, lons, lats


def create_multi_time_comparison(model, times, start_time, output_path):
    """
    Create a multi-panel comparison of age grids at different times.

    Parameters
    ----------
    model : SeafloorAgeModel
        The model instance
    times : list
        List of times to compare
    start_time : float
        Starting time for all evolutions
    output_path : Path
        Output file path
    """
    print(f"\nGenerating multi-time comparison for times: {times}")

    n_times = len(times)
    n_cols = min(3, n_times)
    n_rows = (n_times + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, time in enumerate(times):
        row = idx // n_cols
        col = idx % n_cols

        print(f"  Processing {time} Ma...")
        ages, lons, lats = model.compute_age_grid(time=time, start_time=start_time, resolution=1.0)

        ax = fig.add_subplot(gs[row, col])
        LON, LAT = np.meshgrid(lons, lats)

        im = ax.pcolormesh(LON, LAT, ages, cmap='RdYlBu_r', shading='auto', vmin=0, vmax=200)

        ax.set_xlabel('Longitude (°)', fontsize=10)
        ax.set_ylabel('Latitude (°)', fontsize=10)
        ax.set_title(f'{time} Ma', fontsize=12, fontweight='bold')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, alpha=0.2, linestyle='--')

        # Add mini colorbar
        plt.colorbar(im, ax=ax, label='Age (Ma)', fraction=0.046, pad=0.04)

    # Main title
    fig.suptitle('Seafloor Age Evolution Through Geological Time',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")


def create_age_profile_plot(model, times, latitude, start_time, output_path):
    """
    Create age profile along a specific latitude through time.

    Parameters
    ----------
    model : SeafloorAgeModel
        The model instance
    times : list
        List of times
    latitude : float
        Latitude for the profile (e.g., 0 for equator)
    start_time : float
        Starting time
    output_path : Path
        Output file path
    """
    print(f"\nGenerating age profile at latitude {latitude}°...")

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for time in times:
        print(f"  Processing {time} Ma...")
        ages, lons, lats = model.compute_age_grid(time=time, start_time=start_time, resolution=1.0)

        # Find closest latitude index
        lat_idx = np.argmin(np.abs(lats - latitude))

        # Extract age profile
        age_profile = ages[lat_idx, :]

        # Plot
        ax.plot(lons, age_profile, label=f'{time} Ma', linewidth=2, alpha=0.7)

    ax.set_xlabel('Longitude (°)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Seafloor Age (Ma)', fontsize=13, fontweight='bold')
    ax.set_title(f'Seafloor Age Profile at {latitude}° Latitude',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(-180, 180)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")


def create_age_histogram_evolution(model, times, start_time, output_path):
    """
    Create histogram showing age distribution evolution.

    Parameters
    ----------
    model : SeafloorAgeModel
        The model instance
    times : list
        List of times
    start_time : float
        Starting time
    output_path : Path
        Output file path
    """
    print(f"\nGenerating age histogram evolution...")

    fig, axes = plt.subplots(len(times), 1, figsize=(12, 3 * len(times)))
    if len(times) == 1:
        axes = [axes]

    for idx, time in enumerate(times):
        print(f"  Processing {time} Ma...")
        ages, lons, lats = model.compute_age_grid(time=time, start_time=start_time, resolution=1.0)

        valid_ages = ages[~np.isnan(ages)]

        axes[idx].hist(valid_ages, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_xlabel('Seafloor Age (Ma)', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'Age Distribution at {time} Ma', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add statistics
        mean_age = np.mean(valid_ages)
        axes[idx].axvline(mean_age, color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {mean_age:.1f} Ma')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")


def create_combined_boundary_age_plot(model, time, start_time, output_path):
    """
    Create combined plot showing both age grid and plate boundaries.

    Parameters
    ----------
    model : SeafloorAgeModel
        The model instance
    time : float
        Reconstruction time
    start_time : float
        Starting time
    output_path : Path
        Output file path
    """
    print(f"\nGenerating combined boundary-age plot for {time} Ma...")
    from tractec.geometry import XYZ2LatLon

    # Get age grid
    ages, lons, lats = model.compute_age_grid(time=time, start_time=start_time, resolution=1.0)

    # Get boundaries
    ridge_points = model._get_ridges(time)
    subd_points = model._get_subduction(time)

    # Convert to lat/lon
    ridge_lats, ridge_lons = XYZ2LatLon(ridge_points) if len(ridge_points) > 0 else ([], [])
    subd_lats, subd_lons = XYZ2LatLon(subd_points) if len(subd_points) > 0 else ([], [])

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    LON, LAT = np.meshgrid(lons, lats)

    # Plot age grid
    im = ax.pcolormesh(LON, LAT, ages, cmap='RdYlBu_r', shading='auto',
                       vmin=0, vmax=200, alpha=0.8)

    # Overlay boundaries
    if len(ridge_lons) > 0:
        ax.scatter(ridge_lons, ridge_lats, c='red', s=5, alpha=0.9,
                  label='Mid-Ocean Ridges', edgecolors='darkred', linewidths=0.5)
    if len(subd_lons) > 0:
        ax.scatter(subd_lons, subd_lats, c='blue', s=5, alpha=0.9,
                  label='Subduction Zones', edgecolors='darkblue', linewidths=0.5)

    # Styling
    cbar = plt.colorbar(im, ax=ax, label='Seafloor Age (Ma)', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('Longitude (°)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=13, fontweight='bold')
    ax.set_title(f'Seafloor Ages and Plate Boundaries at {time} Ma',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")


################################################################################
#                         Interactive HTML Plots (Optional)                    #
################################################################################

def create_interactive_plot(model, time, start_time, output_path):
    """
    Create an interactive HTML plot using plotly (if available).

    Parameters
    ----------
    model : SeafloorAgeModel
        The model instance
    time : float
        Reconstruction time
    start_time : float
        Starting time
    output_path : Path
        Output file path (should end in .html)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        print(f"\nGenerating interactive plot for {time} Ma...")

        # Get age grid
        ages, lons, lats = model.compute_age_grid(time=time, start_time=start_time, resolution=2.0)

        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=ages,
            x=lons,
            y=lats,
            colorscale='RdYlBu_r',
            colorbar=dict(title='Age (Ma)'),
            hovertemplate='Lon: %{x}°<br>Lat: %{y}°<br>Age: %{z:.1f} Ma<extra></extra>'
        ))

        fig.update_layout(
            title=f'Interactive Seafloor Age Grid at {time} Ma',
            xaxis_title='Longitude (°)',
            yaxis_title='Latitude (°)',
            width=1200,
            height=600,
            hovermode='closest'
        )

        fig.write_html(output_path)
        print(f"  Saved interactive plot: {output_path}")

    except ImportError:
        print("  Plotly not installed. Skipping interactive plot.")
        print("  Install with: pip install plotly")


################################################################################
#                                  Main                                        #
################################################################################

def main():
    """Generate all website visualizations."""

    print("\n" + "=" * 70)
    print("  TracTec: Website Visualization Generator")
    print("=" * 70)

    # Initialize model
    print("\nInitializing model...")
    config = TracerConfig(
        ridge_resolution=50e3,
        subduction_resolution=20e3,
        ridge_offset=1e3
    )

    model = SeafloorAgeModel(
        rotation_files=rotation_files,
        topology_files=plate_boundary_files,
        continental_polygons=continent_polygons,
        config=config
    )

    # Pre-load boundaries
    print("Pre-loading boundaries...")
    model.preload_boundaries(range(0, 201))

    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)

    # 1. High-quality single age grids
    for time in [0, 50, 100, 150, 200]:
        create_high_quality_age_grid(
            model, time, start_time=200,
            output_path=output_dir / f'high_quality_age_grid_{time}Ma.png'
        )

    # 2. Multi-time comparison
    create_multi_time_comparison(
        model, times=[0, 50, 100, 150, 200], start_time=200,
        output_path=output_dir / 'multi_time_comparison.png'
    )

    # 3. Age profile plots
    create_age_profile_plot(
        model, times=[0, 50, 100], latitude=0, start_time=200,
        output_path=output_dir / 'age_profile_equator.png'
    )

    # 4. Histogram evolution
    create_age_histogram_evolution(
        model, times=[0, 100, 200], start_time=200,
        output_path=output_dir / 'age_histogram_evolution.png'
    )

    # 5. Combined boundary-age plots
    for time in [0, 100]:
        create_combined_boundary_age_plot(
            model, time, start_time=200,
            output_path=output_dir / f'combined_boundary_age_{time}Ma.png'
        )

    # 6. Interactive plot (if plotly available)
    create_interactive_plot(
        model, time=100, start_time=200,
        output_path=output_dir / 'interactive_age_grid_100Ma.html'
    )

    # Summary
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"\nAll visualizations saved to: {output_dir}")
    print(f"\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")

    print("\nThese high-resolution plots are ready for website use!")


if __name__ == "__main__":
    from pathlib import Path

    if not Path(folder_path).exists():
        print("\n" + "!" * 70)
        print("ERROR: Data folder not found!")
        print("!" * 70)
        print(f"\nPlease update 'folder_path' at the top of this script.")
        print(f"Current path: {folder_path}")
        exit(1)

    main()
    print("\n✅ All done!\n")
