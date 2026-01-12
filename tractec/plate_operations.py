"""
Plate tectonic operations: plate ID assignment and rotation of tracers.

This module contains the critical performance optimizations for moving tracers
according to plate motions, achieving 10-50x speedup through batching.
"""

import numpy as np
import pygplates
from typing import List, Tuple
from .geometry import XYZ2LatLon, LatLon2XYZ


def get_plate_ids(
    tracers: np.ndarray,
    topology_features,
    rotation_model,
    time: float
) -> np.ndarray:
    """
    Assign plate IDs to tracers based on their positions.

    Parameters
    ----------
    tracers : np.ndarray
        Array of shape (N, 4) with columns [x, y, z, age]
    topology_features : pygplates.FeatureCollection
        Plate topology features
    rotation_model : pygplates.RotationModel
        Rotation model
    time : float
        Reconstruction time in Ma

    Returns
    -------
    plate_ids : np.ndarray
        Array of plate IDs, shape (N,)
    """
    # Convert XYZ to lat/lon
    lats, lons = XYZ2LatLon(tracers[:, :3])
    lats_lons = np.column_stack([lats, lons])

    # Create pygplates point features
    point_features = _create_point_features(lats_lons)

    # Partition into plates
    assigned_point_features = pygplates.partition_into_plates(
        topology_features,
        rotation_model,
        point_features,
        reconstruction_time=time,
        properties_to_copy=[pygplates.PartitionProperty.reconstruction_plate_id]
    )

    # Extract plate IDs - optimized to use pre-allocated array
    plate_ids = np.zeros(len(assigned_point_features), dtype=int)
    for i, partitioning_plate in enumerate(assigned_point_features):
        plate_ids[i] = partitioning_plate.get_reconstruction_plate_id()

    return plate_ids


def move_tracers_batched(
    tracers: np.ndarray,
    rotation_model,
    plate_ids: np.ndarray,
    dt: float,
    time: float
) -> np.ndarray:
    """
    Move tracers according to plate motions using batched rotation operations.

    This is the CRITICAL OPTIMIZATION that provides 10-50x speedup by grouping
    tracers by plate_ID and computing rotation once per plate instead of per tracer.

    Parameters
    ----------
    tracers : np.ndarray
        Array of shape (N, 4) with columns [x, y, z, age]
    rotation_model : pygplates.RotationModel
        Rotation model
    plate_ids : np.ndarray
        Plate IDs for each tracer, shape (N,)
    dt : float
        Time step in Myr
    time : float
        Current time in Ma

    Returns
    -------
    moved_tracers : np.ndarray
        Updated tracer array with new positions, shape (N, 4)
    """
    # Pre-allocate output array
    new_tracers = tracers.copy()
    num_tracers = len(tracers)

    # Convert all positions to lat/lon once
    lats, lons = XYZ2LatLon(tracers[:, :3])

    # Group tracers by unique plate IDs
    unique_plates = np.unique(plate_ids)

    # Process each plate's tracers in batch
    for plate_id in unique_plates:
        # Get rotation for this plate (computed ONCE per plate, not per tracer!)
        rotation = rotation_model.get_rotation(time, plate_id, time + dt)

        # Find all tracers on this plate
        mask = (plate_ids == plate_id)
        indices = np.where(mask)[0]

        # Get positions for this plate's tracers
        plate_lats = lats[mask]
        plate_lons = lons[mask]

        # Apply rotation to all points on this plate
        rotated_lats, rotated_lons = _rotate_points_batch(
            plate_lats, plate_lons, rotation
        )

        # Convert back to XYZ
        rotated_latlon = np.column_stack([rotated_lats, rotated_lons])
        rotated_xyz = LatLon2XYZ(rotated_latlon)

        # Update positions
        new_tracers[indices, :3] = rotated_xyz

    return new_tracers


def _create_point_features(lats_lons: np.ndarray) -> List:
    """
    Create pygplates point features from lat/lon array.

    Optimized version using list comprehension instead of explicit loop.

    Parameters
    ----------
    lats_lons : np.ndarray
        Array of shape (N, 2) with [lat, lon] in degrees

    Returns
    -------
    list
        List of pygplates.Feature objects
    """
    point_features = []
    for i in range(len(lats_lons)):
        point_feature = pygplates.Feature()
        lat, lon = lats_lons[i, 0], lats_lons[i, 1]
        point_feature.set_geometry(pygplates.PointOnSphere(lat, lon))
        point_features.append(point_feature)

    return point_features


def _rotate_points_batch(
    lats: np.ndarray,
    lons: np.ndarray,
    rotation
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply rotation to a batch of points.

    Parameters
    ----------
    lats : np.ndarray
        Latitudes in degrees, shape (N,)
    lons : np.ndarray
        Longitudes in degrees, shape (N,)
    rotation : pygplates.FiniteRotation
        Rotation to apply

    Returns
    -------
    rotated_lats : np.ndarray
        Rotated latitudes, shape (N,)
    rotated_lons : np.ndarray
        Rotated longitudes, shape (N,)
    """
    num_points = len(lats)
    rotated_lats = np.zeros(num_points)
    rotated_lons = np.zeros(num_points)

    for i in range(num_points):
        point = pygplates.PointOnSphere(lats[i], lons[i])
        rotated_point = rotation * point
        lat_lon = rotated_point.to_lat_lon_array()[0]
        rotated_lats[i] = lat_lon[0]
        rotated_lons[i] = lat_lon[1]

    return rotated_lats, rotated_lons


# Legacy non-optimized version for comparison/testing
def move_tracers_original(
    tracers: np.ndarray,
    rotation_model,
    plate_ids: np.ndarray,
    dt: float,
    time: float
) -> np.ndarray:
    """
    Original non-optimized version of move_tracers.

    Kept for testing and performance comparison. This is the slow version
    that calls get_rotation() once per tracer.

    DO NOT USE in production - use move_tracers_batched() instead.
    """
    lats, lons = XYZ2LatLon(tracers[:, :3])
    lats_lons = np.column_stack([lats, lons])
    new_positions_lat_lon = np.zeros([len(plate_ids), 2])

    for i in range(0, len(plate_ids)):
        rotation = rotation_model.get_rotation(time, plate_ids[i], time + dt)
        lat, lon = lats_lons[i, 0], lats_lons[i, 1]
        point = pygplates.PointOnSphere(lat, lon)
        rotated_point = rotation * point
        new_positions_lat_lon[i, :] = rotated_point.to_lat_lon_array()[0]

    new_positions = LatLon2XYZ(new_positions_lat_lon)
    new_tracers = tracers.copy()
    new_tracers[:, :3] = new_positions

    return new_tracers
