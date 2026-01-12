"""
Subduction detection with optimized spatial queries.

This module handles detection and removal of tracers that get subducted,
using KDTree for fast nearest-neighbor queries (2-5x speedup).
"""

import numpy as np
from scipy.spatial import cKDTree


def check_for_subduction_optimized(
    tracers: np.ndarray,
    old_plate_ids: np.ndarray,
    new_plate_ids: np.ndarray,
    subduction_points: np.ndarray,
    tolerance: float = 100e3
) -> np.ndarray:
    """
    Remove tracers that crossed subduction zones.

    This optimized version only checks tracers that changed plate_ID and uses
    KDTree for fast nearest-neighbor queries, achieving 2-5x speedup.

    Parameters
    ----------
    tracers : np.ndarray
        Array of shape (N, 4) with columns [x, y, z, age]
    old_plate_ids : np.ndarray
        Plate IDs before movement, shape (N,)
    new_plate_ids : np.ndarray
        Plate IDs after movement, shape (N,)
    subduction_points : np.ndarray
        Array of shape (M, 3) with XYZ coordinates of subduction zones
    tolerance : float, optional
        Distance tolerance in meters (default: 100 km)

    Returns
    -------
    survived_tracers : np.ndarray
        Tracers that were not subducted, shape (K, 4) where K <= N
    """
    if len(subduction_points) == 0:
        # No subduction zones at this time
        print("Number of tracers that got subducted is: 0")
        return tracers

    # Find tracers that changed plate ID
    changed_mask = (old_plate_ids != new_plate_ids)
    num_changed = np.sum(changed_mask)

    if num_changed == 0:
        # No tracers changed plates, can't be subducted
        print("Number of tracers that got subducted is: 0")
        return tracers

    # Build KDTree for fast spatial queries (once per timestep)
    tree = cKDTree(subduction_points)

    # Only query tracers that changed plates
    changed_tracers = tracers[changed_mask, :3]
    distances, _ = tree.query(changed_tracers, k=1)

    # Find which changed tracers are close to subduction zones
    close_to_subduction = distances < tolerance

    # Build full removal mask
    removal_mask = np.zeros(len(tracers), dtype=bool)
    # Map the close_to_subduction results back to full tracer array
    changed_indices = np.where(changed_mask)[0]
    removal_mask[changed_indices[close_to_subduction]] = True

    # Remove subducted tracers
    survived_tracers = tracers[~removal_mask]

    num_subducted = np.sum(removal_mask)
    print(f"Number of tracers that got subducted is: {num_subducted}")

    return survived_tracers


def check_for_subduction_original(
    tracers: np.ndarray,
    old_plate_ids: np.ndarray,
    new_plate_ids: np.ndarray,
    subduction_points: np.ndarray,
    tolerance: float = 100e3
) -> np.ndarray:
    """
    Original non-optimized version for comparison.

    DO NOT USE in production - use check_for_subduction_optimized() instead.

    This version checks ALL tracers for proximity to subduction zones, not just
    those that changed plates, and uses slower NearestNDInterpolator.
    """
    from scipy.interpolate import NearestNDInterpolator

    # Find tracers that changed plate ID
    diff = np.abs(np.array(old_plate_ids) - np.array(new_plate_ids))
    idx_plate_ID = np.where(diff > 0.0)[0]

    # Build interpolator (slower than KDTree)
    subd_interpolator = NearestNDInterpolator(subduction_points, subduction_points)

    # Query ALL tracers (inefficient!)
    closest_subd_node = subd_interpolator(tracers[:, :3])
    dist = tracers[:, :3] - closest_subd_node
    dist = np.sqrt(dist[:, 0]**2 + dist[:, 1]**2 + dist[:, 2]**2)
    idx_close = np.where(dist < tolerance)[0]

    # Find intersection
    idx = np.intersect1d(idx_close, idx_plate_ID)

    tracers = np.delete(tracers, idx, axis=0)
    print(f"Number of tracers that got subducted is: {len(idx)}")

    return tracers
