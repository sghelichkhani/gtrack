"""
Boundary location management and caching for mid-ocean ridges and subduction zones.

This module handles the computation and caching of plate boundary locations,
eliminating file I/O bottlenecks through intelligent in-memory caching.
"""

import numpy as np
import pygplates
from functools import lru_cache
from typing import Optional, Tuple, Dict
from .geometry import LatLon2XYZ, Segments2Points


class BoundaryCache:
    """
    Cache for ridge and subduction zone locations across multiple timesteps.

    This class pre-computes and stores boundary locations to avoid repeated
    file I/O and topology resolution. Supports both eager pre-loading and
    lazy on-demand computation.

    Parameters
    ----------
    topology_features : pygplates.FeatureCollection
        Plate topology features
    rotation_model : pygplates.RotationModel
        Plate rotation model
    ridge_resolution : float, optional
        Sampling resolution for ridges in meters (default: 50 km)
    subduction_resolution : float, optional
        Sampling resolution for subduction zones in meters (default: 20 km)
    """

    def __init__(
        self,
        topology_features,
        rotation_model,
        ridge_resolution: float = 50e3,
        subduction_resolution: float = 20e3,
    ):
        self.topology_features = topology_features
        self.rotation_model = rotation_model
        self.ridge_resolution = ridge_resolution
        self.subduction_resolution = subduction_resolution

        # Cache dictionaries: {time: xyz_points}
        self._ridge_cache: Dict[float, np.ndarray] = {}
        self._subduction_cache: Dict[float, np.ndarray] = {}

    def preload(self, time_range):
        """
        Pre-compute and cache boundaries for a range of times.

        Parameters
        ----------
        time_range : iterable
            Time points to pre-compute (e.g., range(0, 400))
        """
        for time in time_range:
            # Compute both to populate caches
            _ = self.get_ridges(time)
            _ = self.get_subduction(time)
            print(f"Done preloading boundaries for time: {time}")

    def get_ridges(self, time: float, as_xyz: bool = True) -> np.ndarray:
        """
        Get mid-ocean ridge locations at specified time.

        Parameters
        ----------
        time : float
            Time in Ma
        as_xyz : bool, default=True
            If True, return Cartesian XYZ coordinates.
            If False, return lat/lon coordinates.

        Returns
        -------
        np.ndarray
            Ridge points as (N, 3) XYZ or (N, 2) lat/lon array
        """
        if time not in self._ridge_cache:
            self._ridge_cache[time] = self._compute_ridges(time)

        ridges = self._ridge_cache[time]

        if not as_xyz and ridges.shape[1] == 3:
            # Convert from XYZ to lat/lon
            from .geometry import XYZ2LatLon
            lats, lons = XYZ2LatLon(ridges)
            return np.column_stack([lats, lons])

        return ridges

    def get_subduction(self, time: float, as_xyz: bool = True) -> np.ndarray:
        """
        Get subduction zone locations at specified time.

        Parameters
        ----------
        time : float
            Time in Ma
        as_xyz : bool, default=True
            If True, return Cartesian XYZ coordinates.
            If False, return lat/lon coordinates.

        Returns
        -------
        np.ndarray
            Subduction points as (N, 3) XYZ or (N, 2) lat/lon array
        """
        if time not in self._subduction_cache:
            self._subduction_cache[time] = self._compute_subduction(time)

        subduction = self._subduction_cache[time]

        if not as_xyz and subduction.shape[1] == 3:
            # Convert from XYZ to lat/lon
            from .geometry import XYZ2LatLon
            lats, lons = XYZ2LatLon(subduction)
            return np.column_stack([lats, lons])

        return subduction

    def _compute_ridges(self, time: float) -> np.ndarray:
        """Compute ridge locations at time and convert to XYZ."""
        ridge_segments, _ = _get_ridge_subduction_locations(
            time, self.topology_features, self.rotation_model
        )

        # Convert segments to points at desired resolution
        lats, lons = Segments2Points(ridge_segments, self.ridge_resolution)

        if len(lats) == 0:
            # No ridges at this time
            return np.zeros((0, 3))

        # Convert to XYZ
        latlon = np.column_stack([lats, lons])
        return LatLon2XYZ(latlon)

    def _compute_subduction(self, time: float) -> np.ndarray:
        """Compute subduction zone locations at time and convert to XYZ."""
        _, subduction_segments = _get_ridge_subduction_locations(
            time, self.topology_features, self.rotation_model
        )

        # Convert segments to points at desired resolution
        lats, lons = Segments2Points(subduction_segments, self.subduction_resolution)

        if len(lats) == 0:
            # No subduction zones at this time
            return np.zeros((0, 3))

        # Convert to XYZ
        latlon = np.column_stack([lats, lons])
        return LatLon2XYZ(latlon)

    def clear(self):
        """Clear all cached data."""
        self._ridge_cache.clear()
        self._subduction_cache.clear()

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get approximate memory usage of cached data.

        Returns
        -------
        dict
            Dictionary with 'ridges_mb', 'subduction_mb', and 'total_mb'
        """
        ridge_bytes = sum(arr.nbytes for arr in self._ridge_cache.values())
        subduction_bytes = sum(arr.nbytes for arr in self._subduction_cache.values())

        return {
            'ridges_mb': ridge_bytes / (1024 * 1024),
            'subduction_mb': subduction_bytes / (1024 * 1024),
            'total_mb': (ridge_bytes + subduction_bytes) / (1024 * 1024),
            'num_timesteps_cached': len(self._ridge_cache),
        }


def _get_ridge_subduction_locations(
    time: float,
    topology_features,
    rotation_model
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mid-ocean ridge and subduction zone locations at a given time.

    This is the core function extracted from TracTec.py that resolves
    plate topologies and identifies boundary types.

    Parameters
    ----------
    time : float
        Reconstruction time in Ma
    topology_features : pygplates.FeatureCollection
        Plate topology features
    rotation_model : pygplates.RotationModel
        Rotation model

    Returns
    -------
    ridge_segments : np.ndarray
        Array of shape (N, 4) with [lat0, lon0, lat1, lon1] for ridges
    subduction_segments : np.ndarray
        Array of shape (M, 4) with [lat0, lon0, lat1, lon1] for subduction zones
    """
    # Resolve topologies at the current time
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(
        topology_features,
        rotation_model,
        resolved_topologies,
        time,
        shared_boundary_sections
    )

    # Accumulate ridge and subduction zone locations
    ridge_lats0 = np.array([])
    ridge_lons0 = np.array([])
    ridge_lats1 = np.array([])
    ridge_lons1 = np.array([])
    subduction_lats0 = np.array([])
    subduction_lons0 = np.array([])
    subduction_lats1 = np.array([])
    subduction_lons1 = np.array([])

    # Iterate over shared boundary sections
    for shared_boundary_section in shared_boundary_sections:

        # Check if segment is mid-ocean ridge
        if shared_boundary_section.get_feature().get_feature_type() == \
                pygplates.FeatureType.gpml_mid_ocean_ridge:

            # Iterate over shared sub-segments
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                lat_lon = shared_sub_segment.get_resolved_geometry().to_lat_lon_array()
                lats = lat_lon[:, 0]
                lons = lat_lon[:, 1]

                ridge_lats0 = np.append(ridge_lats0, lats[:-1])
                ridge_lats1 = np.append(ridge_lats1, lats[1:])
                ridge_lons0 = np.append(ridge_lons0, lons[:-1])
                ridge_lons1 = np.append(ridge_lons1, lons[1:])

        # Check if segment is subduction zone
        if shared_boundary_section.get_feature().get_feature_type() == \
                pygplates.FeatureType.gpml_subduction_zone:

            # Iterate over shared sub-segments
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                lat_lon = shared_sub_segment.get_resolved_geometry().to_lat_lon_array()
                lats = lat_lon[:, 0]
                lons = lat_lon[:, 1]

                subduction_lats0 = np.append(subduction_lats0, lats[:-1])
                subduction_lats1 = np.append(subduction_lats1, lats[1:])
                subduction_lons0 = np.append(subduction_lons0, lons[:-1])
                subduction_lons1 = np.append(subduction_lons1, lons[1:])

    # Build segment arrays
    ridge_segments = np.zeros([len(ridge_lons0), 4])
    subduction_segments = np.zeros([len(subduction_lons0), 4])

    if len(ridge_lons0) > 0:
        ridge_segments[:, 0] = ridge_lats0
        ridge_segments[:, 1] = ridge_lons0
        ridge_segments[:, 2] = ridge_lats1
        ridge_segments[:, 3] = ridge_lons1

    if len(subduction_lons0) > 0:
        subduction_segments[:, 0] = subduction_lats0
        subduction_segments[:, 1] = subduction_lons0
        subduction_segments[:, 2] = subduction_lats1
        subduction_segments[:, 3] = subduction_lons1

    return ridge_segments, subduction_segments
