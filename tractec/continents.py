"""
Continental interaction detection with optimized spatial queries.

This module handles detection and removal of tracers that interact with
continental polygons, using KDTree for fast spatial queries.
"""

import numpy as np
import pygplates
from scipy.spatial import cKDTree
from .geometry import XYZ2LatLon, LatLon2XYZ


def tracers_in_continent_optimized(
    cob_features,
    rotation_model,
    tracers: np.ndarray,
    time: float,
    tolerance: float = 150e3
) -> np.ndarray:
    """
    Remove tracers that interact with continental polygons.

    This optimized version uses KDTree for spatial queries (2x speedup).

    Parameters
    ----------
    cob_features : pygplates.FeatureCollection
        Continent-ocean boundary features
    rotation_model : pygplates.RotationModel
        Rotation model
    tracers : np.ndarray
        Array of shape (N, 4) with columns [x, y, z, age]
    time : float
        Reconstruction time in Ma
    tolerance : float, optional
        Proximity tolerance in meters (default: 150 km)

    Returns
    -------
    survived_tracers : np.ndarray
        Tracers that didn't interact with continents, shape (K, 4) where K <= N
    """
    # Convert tracers to lat/lon
    lats, lons = XYZ2LatLon(tracers[:, :3])
    lats_lons = np.column_stack([lats, lons])

    # Create points list for pygplates
    points = [(lat, lon) for lat, lon in lats_lons]
    multi_point = pygplates.MultiPointOnSphere(points)

    # Reconstruct continental boundaries to current time
    reconstructed_COB = []
    pygplates.reconstruct(cob_features, rotation_model, reconstructed_COB, time)

    # Find points inside continental polygons
    points_inside = np.zeros([1, 2])

    for ReconstructedGeometry in reconstructed_COB:
        some_list = []  # store partitioned_geometries
        polygon = ReconstructedGeometry.get_reconstructed_geometry()
        polygon.partition(multi_point, partitioned_geometries_inside=some_list)

        # Append
        for i in range(0, len(some_list)):
            points_inside = np.append(points_inside, some_list[i].to_lat_lon_array(), axis=0)

    tracers_inside = points_inside[1:, :]

    if len(tracers_inside) > 0:
        # Convert to XYZ for distance calculations
        tracers_xyz = LatLon2XYZ(lats_lons)
        cont_tracers_xyz = LatLon2XYZ(tracers_inside)

        # Use KDTree for fast nearest-neighbor queries
        tree = cKDTree(cont_tracers_xyz)
        distances, _ = tree.query(tracers_xyz, k=1)

        # Find tracers close to continents
        close_idx = np.where(distances < tolerance)[0]

        # Remove these tracers
        tracers = np.delete(tracers, close_idx, axis=0)
        print(f"Number of tracers that interacted with a continent is: {len(close_idx)}")
    else:
        print("Number of tracers that interacted with a continent is: 0")

    return tracers


def tracers_in_continent_original(
    cob_features,
    rotation_model,
    tracers: np.ndarray,
    time: float,
    tolerance: float = 150e3
) -> np.ndarray:
    """
    Original non-optimized version for comparison.

    DO NOT USE in production - use tracers_in_continent_optimized() instead.

    This version uses slower NearestNDInterpolator instead of KDTree.
    """
    from scipy.interpolate import NearestNDInterpolator

    lats, lons = XYZ2LatLon(tracers[:, :3])
    lats_lons = np.column_stack([lats, lons])

    points = []
    for i in range(0, len(lats_lons)):
        lat, lon = lats_lons[i, 0], lats_lons[i, 1]
        points.append((lat, lon))

    points_inside = np.zeros([1, 2])
    multi_point = pygplates.MultiPointOnSphere(points)

    reconstructed_COB = []
    pygplates.reconstruct(cob_features, rotation_model, reconstructed_COB, time)

    for ReconstructedGeometry in reconstructed_COB:
        some_list = []
        polygon = ReconstructedGeometry.get_reconstructed_geometry()
        polygon.partition(multi_point, partitioned_geometries_inside=some_list)

        for i in range(0, len(some_list)):
            points_inside = np.append(points_inside, some_list[i].to_lat_lon_array(), axis=0)

    tracers_inside = points_inside[1:, :]

    if len(tracers_inside) > 0:
        tracers_xyz = LatLon2XYZ(lats_lons)
        cont_tracers_xyz = LatLon2XYZ(tracers_inside)
        tol = tolerance

        # Use slower interpolator
        cont_interpolator = NearestNDInterpolator(cont_tracers_xyz, cont_tracers_xyz)
        closest_cont_tracer = cont_interpolator(tracers_xyz)
        dist = tracers_xyz - closest_cont_tracer
        dist = np.sqrt(dist[:, 0]**2 + dist[:, 1]**2 + dist[:, 2]**2)
        close_idx = np.where(dist < tol)[0]

        tracers = np.delete(tracers, close_idx, axis=0)
        print(f"Number of tracers that interacted with a continent is: {len(close_idx)}")
    else:
        print("Number of tracers that interacted with a continent is: 0")

    return tracers
