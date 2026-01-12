"""
Main SeafloorAgeModel API for on-the-fly seafloor age grid generation.

This module provides the high-level interface for computing seafloor ages
using Lagrangian particle tracking with optimized operations.
"""

import numpy as np
import pygplates
from typing import Optional, Tuple, List
from .config import TracerConfig
from .boundaries import BoundaryCache
from .plate_operations import get_plate_ids, move_tracers_batched
from .ridges import add_ridge_tracers_vectorized
from .subduction import check_for_subduction_optimized
from .continents import tracers_in_continent_optimized


class SeafloorAgeModel:
    """
    Main API for seafloor age computation using plate tectonic reconstructions.

    This class orchestrates the full workflow of tracer-based seafloor age
    computation with performance optimizations for on-the-fly generation.

    Parameters
    ----------
    rotation_files : list of str
        Paths to rotation model files (.rot)
    topology_files : list of str
        Paths to topology/plate boundary files (.gpmlz)
    continental_polygons : str
        Path to continental polygon file (.gpmlz)
    config : TracerConfig, optional
        Configuration parameters. If None, uses defaults.

    Examples
    --------
    >>> model = SeafloorAgeModel(
    ...     rotation_files=['rotations.rot'],
    ...     topology_files=['topologies.gpmlz'],
    ...     continental_polygons='continents.gpmlz'
    ... )
    >>> ages, lons, lats = model.compute_age_grid(time=100, start_time=200)
    """

    def __init__(
        self,
        rotation_files: List[str],
        topology_files: List[str],
        continental_polygons: str,
        config: Optional[TracerConfig] = None
    ):
        """Initialize the seafloor age model."""
        self.rotation_model = pygplates.RotationModel(rotation_files)

        # Load topology features
        self.topology_features = pygplates.FeatureCollection()
        for file in topology_files:
            topology_feature = pygplates.FeatureCollection(file)
            self.topology_features.add(topology_feature)

        self.cob_features = pygplates.FeatureCollection(continental_polygons)

        self.config = config if config is not None else TracerConfig()

        # Boundary cache (initialized on demand)
        self._boundary_cache: Optional[BoundaryCache] = None

    def preload_boundaries(self, time_range):
        """
        Pre-compute and cache ridge/subduction locations for better performance.

        Parameters
        ----------
        time_range : iterable
            Time points to pre-cache (e.g., range(0, 400))

        Examples
        --------
        >>> model.preload_boundaries(range(0, 401))
        """
        if self._boundary_cache is None:
            self._boundary_cache = BoundaryCache(
                self.topology_features,
                self.rotation_model,
                self.config.ridge_resolution,
                self.config.subduction_resolution
            )
        self._boundary_cache.preload(time_range)

    def compute_age_grid(
        self,
        time: float,
        start_time: Optional[float] = None,
        initial_tracers: Optional[np.ndarray] = None,
        resolution: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute seafloor age grid at specified time.

        Parameters
        ----------
        time : float
            Target time in Ma
        start_time : float, optional
            Starting time. If None and no initial_tracers, uses time+1
        initial_tracers : np.ndarray, optional
            Initial tracer array (N, 4). If None, generates from ridges
        resolution : float, optional
            Grid resolution in degrees (default: 0.5)

        Returns
        -------
        ages : np.ndarray
            2D array of seafloor ages
        lons : np.ndarray
            Longitude coordinates
        lats : np.ndarray
            Latitude coordinates

        Examples
        --------
        >>> ages, lons, lats = model.compute_age_grid(time=100, start_time=200)
        """
        # Get tracers at target time
        tracers = self.get_tracers_at_time(time, start_time, initial_tracers)

        # Convert to age grid
        return self._tracers_to_grid(tracers, time, resolution)

    def get_tracers_at_time(
        self,
        time: float,
        start_time: Optional[float] = None,
        initial_tracers: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get tracer positions and ages at specified time.

        Parameters
        ----------
        time : float
            Target time in Ma
        start_time : float, optional
            Starting time. If None and no initial_tracers, uses time+1
        initial_tracers : np.ndarray, optional
            Initial tracer array (N, 4). If None, generates from ridges

        Returns
        -------
        tracers : np.ndarray
            Tracer array (N, 4) with columns [x, y, z, age]

        Examples
        --------
        >>> tracers = model.get_tracers_at_time(time=100, start_time=150)
        """
        # Determine initial state
        if initial_tracers is None:
            if start_time is None:
                start_time = time + 1
            tracers = self._initialize_tracers(start_time)
        else:
            tracers = initial_tracers.copy()
            if start_time is None:
                # Infer start time from tracer ages
                start_time = time + 1

        # Evolve tracers to target time
        tracers = self._evolve_tracers(tracers, start_time, time)

        return tracers

    def _initialize_tracers(self, time: float) -> np.ndarray:
        """Initialize tracers at ridges for given time."""
        # Get ridge locations
        ridge_points = self._get_ridges(time)

        if len(ridge_points) == 0:
            # No ridges, return empty tracer array
            return np.zeros((0, 4))

        # Generate tracers at ridges
        tracers = add_ridge_tracers_vectorized(
            ridge_points,
            self.config.ridge_offset,
            self.config.ridge_resolution
        )

        return tracers

    def _evolve_tracers(
        self,
        tracers: np.ndarray,
        from_time: float,
        to_time: float
    ) -> np.ndarray:
        """
        Evolve tracers through time using plate motions.

        This is the main simulation loop that moves tracers according to
        plate tectonics, removes subducted/continental tracers, and adds
        new tracers at ridges.
        """
        dt = self.config.time_step
        time_direction = -1 if from_time > to_time else 1
        current_time = from_time

        print(f"\nEvolving tracers from {from_time} Ma to {to_time} Ma")
        print(f"Initial tracer count: {len(tracers)}")

        while (time_direction > 0 and current_time < to_time) or \
              (time_direction < 0 and current_time > to_time):

            print(f"\n--- Time step: {current_time} Ma ---")
            print(f"Tracer count: {len(tracers)}")

            # Get plate IDs at current position
            print("Assigning plate IDs to tracers...")
            plate_ids_t0 = get_plate_ids(
                tracers, self.topology_features,
                self.rotation_model, current_time
            )

            # Move tracers
            print("Moving tracers...")
            tracers = move_tracers_batched(
                tracers, self.rotation_model,
                plate_ids_t0, dt * time_direction, current_time
            )

            # Get new plate IDs
            next_time = current_time + dt * time_direction
            print("Assigning plate IDs to next tracer positions...")
            plate_ids_t1 = get_plate_ids(
                tracers, self.topology_features,
                self.rotation_model, next_time
            )

            # Check for subduction
            print("Checking for subduction of tracers...")
            subduction_points = self._get_subduction(next_time)
            tracers = check_for_subduction_optimized(
                tracers, plate_ids_t0, plate_ids_t1,
                subduction_points, self.config.subduction_tolerance
            )

            # Check for continental interaction
            print("Checking for tracer-continent interaction...")
            tracers = tracers_in_continent_optimized(
                self.cob_features, self.rotation_model,
                tracers, next_time, self.config.continental_tolerance
            )

            # Increase age of surviving tracers
            tracers[:, 3] += dt

            # Add new tracers at ridges
            if time_direction < 0 and next_time > 0:
                ridge_points = self._get_ridges(next_time)
                if len(ridge_points) > 0:
                    new_tracers = add_ridge_tracers_vectorized(
                        ridge_points,
                        self.config.ridge_offset,
                        self.config.ridge_resolution
                    )
                    tracers = np.vstack([tracers, new_tracers])
                    print(f"Added {len(new_tracers)} new tracers at ridges")

            current_time = next_time

        print(f"\nFinal tracer count: {len(tracers)}")
        return tracers

    def _get_ridges(self, time: float) -> np.ndarray:
        """Get ridge locations at specified time."""
        if self._boundary_cache is None:
            # Create cache on demand
            self._boundary_cache = BoundaryCache(
                self.topology_features,
                self.rotation_model,
                self.config.ridge_resolution,
                self.config.subduction_resolution
            )
        return self._boundary_cache.get_ridges(time, as_xyz=True)

    def _get_subduction(self, time: float) -> np.ndarray:
        """Get subduction zone locations at specified time."""
        if self._boundary_cache is None:
            self._boundary_cache = BoundaryCache(
                self.topology_features,
                self.rotation_model,
                self.config.ridge_resolution,
                self.config.subduction_resolution
            )
        return self._boundary_cache.get_subduction(time, as_xyz=True)

    def _tracers_to_grid(
        self,
        tracers: np.ndarray,
        time: float,
        resolution: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert tracers to a regular age grid.

        This is a simple implementation for the MVP. Can be enhanced with
        better interpolation methods later.
        """
        from .geometry import XYZ2LatLon

        # Add zero-age tracers at current ridges for better coverage
        ridge_points = self._get_ridges(time)
        if len(ridge_points) > 0:
            ridge_tracers = np.zeros((len(ridge_points), 4))
            ridge_tracers[:, :3] = ridge_points
            ridge_tracers[:, 3] = 0
            all_tracers = np.vstack([tracers, ridge_tracers])
        else:
            all_tracers = tracers

        # Convert to lat/lon
        lats, lons = XYZ2LatLon(all_tracers[:, :3])
        ages = all_tracers[:, 3]

        # Create grid
        lon_bins = np.arange(-180, 180 + resolution, resolution)
        lat_bins = np.arange(-90, 90 + resolution, resolution)

        # Simple nearest-neighbor gridding for MVP
        # (can be replaced with more sophisticated methods)
        age_grid = np.full((len(lat_bins) - 1, len(lon_bins) - 1), np.nan)

        for i in range(len(lats)):
            lat_idx = int((lats[i] + 90) / resolution)
            lon_idx = int((lons[i] + 180) / resolution)

            if 0 <= lat_idx < age_grid.shape[0] and 0 <= lon_idx < age_grid.shape[1]:
                if np.isnan(age_grid[lat_idx, lon_idx]):
                    age_grid[lat_idx, lon_idx] = ages[i]
                else:
                    # Average if multiple tracers in same cell
                    age_grid[lat_idx, lon_idx] = (age_grid[lat_idx, lon_idx] + ages[i]) / 2

        # Grid coordinates
        lon_coords = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_coords = (lat_bins[:-1] + lat_bins[1:]) / 2

        return age_grid, lon_coords, lat_coords
