# Interpolate Gridded Data to Sphere Mesh
# ========================================
#
# This example demonstrates how to interpolate gridded data (e.g., from a
# NetCDF file) onto a sphere mesh using inverse distance weighting (IDW).
#
# **Why use a sphere mesh?**
#
# Regular lat/lon grids have uneven point density - points cluster near the poles.
# A Fibonacci spiral mesh provides approximately uniform spacing across the sphere,
# which is ideal for numerical simulations (like gadopt) that use cKDTree
# interpolation.
#
# **How it works:**
#
# 1. Load gridded data from NetCDF/HDF5 file
# 2. Create sphere mesh with desired number of points (Fibonacci spiral)
# 3. Use cKDTree to find nearest neighbors on the grid for each mesh point
# 4. Apply inverse distance weighting (IDW) to interpolate values
# 5. Save to HDF5 for use in other applications

# +
from pathlib import Path
import argparse
import numpy as np
import h5py
from scipy.spatial import cKDTree

from gtrack.mesh import create_sphere_mesh_xyz
from gtrack.geometry import (
    LatLon2XYZ, XYZ2LatLon, EARTH_RADIUS,
    inverse_distance_weighted_interpolation
)
# -

# ## Configuration
#
# Set defaults that can be overridden via command-line arguments.

# +
# Default paths - update these for your data
DEFAULT_INPUT = Path("./lithospheric_thickness_maps/global/SL2013sv/SL2013sv.nc")
DEFAULT_OUTPUT = Path("./output/lithospheric_thickness_mesh.h5")

# Number of points on the sphere mesh
# 10,000 points: ~110 km average spacing
# 40,000 points: ~55 km average spacing
# 160,000 points: ~28 km average spacing
DEFAULT_NPOINTS = 40000

# Number of neighbors for IDW interpolation
DEFAULT_K_NEIGHBORS = 4
# -


def load_netcdf_grid(filepath):
    """
    Load gridded data from a NetCDF/HDF5 file.

    Expects the file to have:
    - 'lon': longitude array (degrees, 0-360 or -180 to 180)
    - 'lat': latitude array (degrees, -90 to 90)
    - 'z': data values on the grid

    Parameters
    ----------
    filepath : str or Path
        Path to the NetCDF file.

    Returns
    -------
    lon_grid : np.ndarray
        Longitude values for each grid point, shape (N,).
    lat_grid : np.ndarray
        Latitude values for each grid point, shape (N,).
    values : np.ndarray
        Data values at each grid point, shape (N,).
    metadata : dict
        Metadata about the loaded data.
    """
    filepath = Path(filepath)

    with h5py.File(filepath, 'r') as f:
        lon_data = f['lon'][:]  # May be 0-360
        lat_data = f['lat'][:]  # -90 to 90
        z_data = f['z'][:]      # 2D array (lat, lon)

    # Convert longitude from 0-360 to -180 to 180 if needed
    lon_data = np.where(lon_data > 180, lon_data - 360, lon_data)

    # Create meshgrid and flatten
    lon_grid, lat_grid = np.meshgrid(lon_data, lat_data)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()
    values_flat = z_data.ravel()

    metadata = {
        'source_file': str(filepath),
        'n_lon': len(lon_data),
        'n_lat': len(lat_data),
        'lon_range': (float(lon_data.min()), float(lon_data.max())),
        'lat_range': (float(lat_data.min()), float(lat_data.max())),
        'value_range': (float(np.nanmin(values_flat)), float(np.nanmax(values_flat))),
    }

    return lon_flat, lat_flat, values_flat, metadata


def interpolate_to_mesh(
    lon_grid, lat_grid, values,
    n_points=40000,
    k_neighbors=4
):
    """
    Interpolate gridded data to a sphere mesh using IDW.

    Parameters
    ----------
    lon_grid : np.ndarray
        Longitude of grid points (degrees), shape (N,).
    lat_grid : np.ndarray
        Latitude of grid points (degrees), shape (N,).
    values : np.ndarray
        Data values at grid points, shape (N,).
    n_points : int, default=40000
        Number of points on the sphere mesh.
    k_neighbors : int, default=4
        Number of nearest neighbors for IDW interpolation.

    Returns
    -------
    mesh_xyz : np.ndarray
        XYZ coordinates of mesh points, shape (M, 3).
    mesh_values : np.ndarray
        Interpolated values at mesh points, shape (M,).
    """
    # Create sphere mesh (unit sphere coordinates)
    mesh_xyz_unit = create_sphere_mesh_xyz(n_points, radius=1.0)

    print(f"Created sphere mesh with {n_points:,} points")

    # Convert grid lat/lon to XYZ on unit sphere for cKDTree
    latlon_grid = np.column_stack([lat_grid, lon_grid])
    grid_xyz = LatLon2XYZ(latlon_grid)
    # Normalize to unit sphere for consistent distance computation
    grid_xyz_unit = grid_xyz / EARTH_RADIUS

    # Build cKDTree from grid points
    print("Building cKDTree from grid data...")
    tree = cKDTree(grid_xyz_unit)

    # Query k nearest neighbors for each mesh point
    print(f"Querying {k_neighbors} nearest neighbors for each mesh point...")
    distances, indices = tree.query(mesh_xyz_unit, k=k_neighbors)

    # Handle NaN values in the source data
    # Replace NaN with a large value that will have low weight
    values_clean = np.where(np.isnan(values), 0.0, values)
    is_nan = np.isnan(values)

    # Get values from neighbors
    neighbor_values = values_clean[indices]  # Shape: (n_mesh, k_neighbors)
    neighbor_is_nan = is_nan[indices]

    # Apply IDW interpolation
    print("Applying inverse distance weighting...")
    mesh_values = inverse_distance_weighted_interpolation(neighbor_values, distances)

    # Mark points where all neighbors were NaN as NaN
    all_nan_mask = np.all(neighbor_is_nan, axis=1)
    mesh_values[all_nan_mask] = np.nan

    # Scale mesh to Earth radius for output
    mesh_xyz = mesh_xyz_unit * EARTH_RADIUS

    return mesh_xyz, mesh_values


def save_to_hdf5(filepath, mesh_xyz, mesh_values, metadata=None):
    """
    Save interpolated data to HDF5 file.

    The file contains:
    - 'xyz': Cartesian coordinates (N, 3) in meters
    - 'lonlat': Geographic coordinates (N, 2) in degrees [lon, lat]
    - 'values': Interpolated data values (N,)
    - Metadata as attributes

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    mesh_xyz : np.ndarray
        XYZ coordinates, shape (N, 3).
    mesh_values : np.ndarray
        Data values, shape (N,).
    metadata : dict, optional
        Metadata to store as attributes.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Compute lon/lat from XYZ
    lats, lons = XYZ2LatLon(mesh_xyz)
    lonlat = np.column_stack([lons, lats])

    with h5py.File(filepath, 'w') as f:
        # Store coordinates and values
        f.create_dataset('xyz', data=mesh_xyz, compression='gzip')
        f.create_dataset('lonlat', data=lonlat, compression='gzip')
        f.create_dataset('values', data=mesh_values, compression='gzip')

        # Store metadata as attributes
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (tuple, list)):
                    f.attrs[key] = np.array(value)
                else:
                    f.attrs[key] = value

        # Always store basic info
        f.attrs['n_points'] = len(mesh_xyz)
        f.attrs['coordinate_system'] = 'Cartesian XYZ in meters, Earth radius = 6.3781e6 m'

    print(f"Saved to {filepath}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Interpolate gridded data to sphere mesh using IDW.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (40,000 points)
  python interpolate_to_mesh.py

  # Specify input/output files
  python interpolate_to_mesh.py -i data.nc -o output.h5

  # Higher resolution mesh
  python interpolate_to_mesh.py -n 160000

  # More neighbors for smoother interpolation
  python interpolate_to_mesh.py -k 8
"""
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default=DEFAULT_INPUT,
        help=f'Input NetCDF file (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Output HDF5 file (default: {DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '-n', '--npoints',
        type=int,
        default=DEFAULT_NPOINTS,
        help=f'Number of mesh points (default: {DEFAULT_NPOINTS})'
    )
    parser.add_argument(
        '-k', '--neighbors',
        type=int,
        default=DEFAULT_K_NEIGHBORS,
        help=f'Number of neighbors for IDW (default: {DEFAULT_K_NEIGHBORS})'
    )
    parser.add_argument(
        '--variable',
        type=str,
        default='z',
        help='Variable name in NetCDF file (default: z)'
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("Interpolate Gridded Data to Sphere Mesh")
    print("=" * 60)
    print(f"Input file:        {args.input}")
    print(f"Output file:       {args.output}")
    print(f"Mesh points:       {args.npoints:,}")
    print(f"IDW neighbors:     {args.neighbors}")
    print("=" * 60)

    # Load gridded data
    print("\nLoading gridded data...")
    lon_grid, lat_grid, values, metadata = load_netcdf_grid(args.input)
    print(f"  Grid size: {metadata['n_lon']} x {metadata['n_lat']} = {len(values):,} points")
    print(f"  Value range: {metadata['value_range'][0]:.2f} to {metadata['value_range'][1]:.2f}")

    # Interpolate to sphere mesh
    print("\nInterpolating to sphere mesh...")
    mesh_xyz, mesh_values = interpolate_to_mesh(
        lon_grid, lat_grid, values,
        n_points=args.npoints,
        k_neighbors=args.neighbors
    )

    # Update metadata
    metadata['n_points'] = args.npoints
    metadata['k_neighbors'] = args.neighbors
    metadata['interpolated_range'] = (float(np.nanmin(mesh_values)), float(np.nanmax(mesh_values)))

    # Save to HDF5
    print("\nSaving to HDF5...")
    save_to_hdf5(args.output, mesh_xyz, mesh_values, metadata)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Input points:      {len(values):,}")
    print(f"Output points:     {len(mesh_xyz):,}")
    print(f"Original range:    {metadata['value_range'][0]:.2f} - {metadata['value_range'][1]:.2f}")
    print(f"Interpolated range: {metadata['interpolated_range'][0]:.2f} - {metadata['interpolated_range'][1]:.2f}")
    n_nan = np.sum(np.isnan(mesh_values))
    if n_nan > 0:
        print(f"NaN values:        {n_nan:,} ({100*n_nan/len(mesh_values):.1f}%)")
    print("=" * 60)


# ## Run as Script or Interactive
#
# When run as a script, uses command-line arguments.
# When imported or run interactively, you can call the functions directly.

if __name__ == '__main__':
    main()
# -

# ## Example: Interactive Usage
#
# You can also use the functions directly in Python:
#
# ```python
# from interpolate_to_mesh import (
#     load_netcdf_grid,
#     interpolate_to_mesh,
#     save_to_hdf5
# )
#
# # Load data
# lon, lat, values, meta = load_netcdf_grid('my_data.nc')
#
# # Interpolate with custom settings
# xyz, interp_values = interpolate_to_mesh(
#     lon, lat, values,
#     n_points=160000,  # Higher resolution
#     k_neighbors=8     # More neighbors for smoother result
# )
#
# # Save
# save_to_hdf5('output.h5', xyz, interp_values, meta)
# ```
#
# ## Reading the Output
#
# The HDF5 file can be read with:
#
# ```python
# import h5py
# import numpy as np
#
# with h5py.File('output.h5', 'r') as f:
#     xyz = f['xyz'][:]           # (N, 3) Cartesian coordinates
#     lonlat = f['lonlat'][:]     # (N, 2) [lon, lat] in degrees
#     values = f['values'][:]     # (N,) interpolated values
#
#     # Access metadata
#     n_points = f.attrs['n_points']
# ```
