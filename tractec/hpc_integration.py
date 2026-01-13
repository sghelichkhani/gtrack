"""
HPC-optimized seafloor age tracking for forward-time simulations.

This module provides a stateful interface for integrating seafloor age
calculations into HPC geodynamic simulations that run forward in time.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .model import SeafloorAgeModel
from .config import TracerConfig
from .geometry import XYZ2LatLon


class HPCSeafloorAgeTracker:
    """
    Stateful seafloor age tracker optimized for HPC forward-time simulations.

    This class maintains tracer state in memory and provides incremental updates
    as a simulation progresses forward in time. Designed for minimal overhead
    in HPC workflows.

    Key features:
    - Stateful: Maintains tracers in memory between updates
    - Incremental: Only evolves from last state to new state
    - Efficient: Pre-loads all boundaries, minimal I/O
    - HPC-friendly: Returns raw arrays, no plotting overhead

    Example Usage
    -------------
    >>> # Initialize at start of simulation
    >>> tracker = HPCSeafloorAgeTracker(
    ...     rotation_files=['rotations.rot'],
    ...     topology_files=['topologies.gpmlz'],
    ...     continental_polygons='continents.gpmlz',
    ...     initial_time=200,
    ...     max_time=0
    ... )
    >>>
    >>> # Initialize tracers
    >>> tracker.initialize_at_time(200)
    >>>
    >>> # In simulation loop, update as needed
    >>> age_data = tracker.update_to_time(195)  # Forward 5 Myr
    >>> lons, lats, ages = age_data['lons'], age_data['lats'], age_data['ages']
    >>>
    >>> # Later in simulation
    >>> age_data = tracker.update_to_time(180)  # Incremental update
    """

    def __init__(
        self,
        rotation_files: list,
        topology_files: list,
        continental_polygons: str,
        initial_time: float,
        max_time: float = 0,
        config: Optional[TracerConfig] = None,
        preload_boundaries: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the HPC seafloor age tracker.

        Parameters
        ----------
        rotation_files : list of str
            Paths to rotation model files
        topology_files : list of str
            Paths to topology files
        continental_polygons : str
            Path to continental polygon file
        initial_time : float
            Starting time for simulation (e.g., 200 Ma)
        max_time : float, default=0
            Maximum time to simulate to (e.g., 0 Ma for present)
        config : TracerConfig, optional
            Configuration parameters. If None, uses defaults.
        preload_boundaries : bool, default=True
            Whether to pre-load all boundaries. Set False for memory-constrained systems.
        verbose : bool, default=True
            Print progress information
        """
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing HPC Seafloor Age Tracker...")
            print(f"  Time range: {initial_time} Ma → {max_time} Ma")

        # Initialize TracTec model
        self.model = SeafloorAgeModel(
            rotation_files=rotation_files,
            topology_files=topology_files,
            continental_polygons=continental_polygons,
            config=config if config else TracerConfig()
        )

        # Pre-load boundaries if requested
        if preload_boundaries:
            if self.verbose:
                print(f"  Pre-loading boundaries...")
            self.model.preload_boundaries(range(max_time, int(initial_time) + 1))

            if self.verbose:
                memory_info = self.model._boundary_cache.get_memory_usage()
                print(f"  Cached {memory_info['num_timesteps_cached']} timesteps")
                print(f"  Memory: {memory_info['total_mb']:.1f} MB")

        # State variables
        self.current_time = initial_time
        self.initial_time = initial_time
        self.max_time = max_time
        self.current_tracers = None
        self._initialized = False

        if self.verbose:
            print("  Initialization complete.")

    def initialize_at_time(self, time: float) -> int:
        """
        Initialize tracers at starting time.

        Call this once at the beginning of your simulation.

        Parameters
        ----------
        time : float
            Starting time (e.g., 200 Ma)

        Returns
        -------
        int
            Number of tracers initialized

        Examples
        --------
        >>> tracker.initialize_at_time(200)
        Initialized with 12534 tracers
        12534
        """
        if self.verbose:
            print(f"\nInitializing tracers at {time} Ma...")

        self.current_tracers = self.model._initialize_tracers(time)
        self.current_time = time
        self._initialized = True

        num_tracers = len(self.current_tracers)

        if self.verbose:
            print(f"  Initialized with {num_tracers} tracers")

        return num_tracers

    def update_to_time(self, new_time: float) -> Dict:
        """
        Update seafloor ages from current time to new time.

        This is the main method to call when plate configuration changes
        in your simulation. It incrementally evolves tracers from the
        last known state.

        Parameters
        ----------
        new_time : float
            New simulation time in Ma

        Returns
        -------
        dict
            Dictionary with keys:
            - 'lons': np.ndarray of longitudes
            - 'lats': np.ndarray of latitudes
            - 'ages': np.ndarray of seafloor ages
            - 'xyz': np.ndarray of Cartesian coordinates (N, 3)
            - 'count': int, number of tracers
            - 'time': float, current time

        Raises
        ------
        RuntimeError
            If initialize_at_time() has not been called

        Examples
        --------
        >>> age_data = tracker.update_to_time(195)
        >>> print(f"Ages range from {age_data['ages'].min():.1f} to {age_data['ages'].max():.1f} Ma")
        """
        if not self._initialized:
            raise RuntimeError(
                "Must call initialize_at_time() before update_to_time()!"
            )

        if new_time == self.current_time:
            # No update needed
            if self.verbose:
                print(f"Already at {new_time} Ma, no update needed")
            return self._get_current_state()

        if self.verbose:
            print(f"\nUpdating ages: {self.current_time} Ma → {new_time} Ma")

        # Evolve tracers incrementally
        self.current_tracers = self.model._evolve_tracers(
            self.current_tracers,
            from_time=self.current_time,
            to_time=new_time
        )

        self.current_time = new_time

        if self.verbose:
            print(f"  Current tracer count: {len(self.current_tracers)}")

        return self._get_current_state()

    def get_current_ages(self) -> Dict:
        """
        Get current seafloor ages without updating.

        Returns current state in the same format as update_to_time().

        Returns
        -------
        dict
            Current age data

        Examples
        --------
        >>> current = tracker.get_current_ages()
        >>> print(f"At {current['time']} Ma: {current['count']} tracers")
        """
        return self._get_current_state()

    def _get_current_state(self) -> Dict:
        """Internal method to package current state."""
        if self.current_tracers is None or len(self.current_tracers) == 0:
            return {
                'lons': np.array([]),
                'lats': np.array([]),
                'ages': np.array([]),
                'xyz': np.zeros((0, 3)),
                'count': 0,
                'time': self.current_time
            }

        # Convert to lat/lon
        lats, lons = XYZ2LatLon(self.current_tracers[:, :3])
        ages = self.current_tracers[:, 3]

        return {
            'lons': lons,
            'lats': lats,
            'ages': ages,
            'xyz': self.current_tracers[:, :3].copy(),
            'count': len(ages),
            'time': self.current_time
        }

    def save_checkpoint(self, filename: str):
        """
        Save current tracer state to file for checkpointing.

        Parameters
        ----------
        filename : str
            Path to save checkpoint file (.npy format)

        Examples
        --------
        >>> tracker.save_checkpoint('checkpoint_150Ma.npy')
        """
        if not self._initialized:
            raise RuntimeError("No state to save - not initialized")

        checkpoint_data = {
            'tracers': self.current_tracers,
            'time': self.current_time,
            'initial_time': self.initial_time,
            'max_time': self.max_time
        }

        np.save(filename, checkpoint_data, allow_pickle=True)

        if self.verbose:
            print(f"Saved checkpoint to {filename} ({self.current_time} Ma)")

    def load_checkpoint(self, filename: str):
        """
        Load tracer state from checkpoint file.

        Parameters
        ----------
        filename : str
            Path to checkpoint file

        Examples
        --------
        >>> tracker.load_checkpoint('checkpoint_150Ma.npy')
        """
        checkpoint_data = np.load(filename, allow_pickle=True).item()

        self.current_tracers = checkpoint_data['tracers']
        self.current_time = checkpoint_data['time']
        self.initial_time = checkpoint_data['initial_time']
        self.max_time = checkpoint_data['max_time']
        self._initialized = True

        if self.verbose:
            print(f"Loaded checkpoint from {filename}")
            print(f"  Time: {self.current_time} Ma")
            print(f"  Tracers: {len(self.current_tracers)}")

    def get_statistics(self) -> Dict:
        """
        Get statistics about current tracer state.

        Returns
        -------
        dict
            Statistics including mean age, max age, coverage, etc.
        """
        if not self._initialized or len(self.current_tracers) == 0:
            return {
                'count': 0,
                'mean_age': 0,
                'max_age': 0,
                'min_age': 0,
                'time': self.current_time
            }

        ages = self.current_tracers[:, 3]

        return {
            'count': len(ages),
            'mean_age': np.mean(ages),
            'max_age': np.max(ages),
            'min_age': np.min(ages),
            'std_age': np.std(ages),
            'time': self.current_time
        }


class MemoryEfficientSeafloorAgeTracker:
    """
    Memory-efficient variant that computes ages on-demand.

    This class does not maintain state between calls. Use when memory
    is constrained or updates are infrequent.

    Example Usage
    -------------
    >>> tracker = MemoryEfficientSeafloorAgeTracker(...)
    >>>
    >>> # Compute ages on demand (no state)
    >>> age_data = tracker.get_ages_at_time(time=100, start_time=200)
    """

    def __init__(
        self,
        rotation_files: list,
        topology_files: list,
        continental_polygons: str,
        config: Optional[TracerConfig] = None
    ):
        """Initialize memory-efficient tracker."""
        self.model = SeafloorAgeModel(
            rotation_files=rotation_files,
            topology_files=topology_files,
            continental_polygons=continental_polygons,
            config=config if config else TracerConfig()
        )

    def get_ages_at_time(
        self,
        time: float,
        start_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ages at specific time without maintaining state.

        Parameters
        ----------
        time : float
            Target time
        start_time : float
            Starting time for computation

        Returns
        -------
        lons : np.ndarray
            Longitudes
        lats : np.ndarray
            Latitudes
        ages : np.ndarray
            Seafloor ages
        """
        tracers = self.model.get_tracers_at_time(
            time=time,
            start_time=start_time
        )

        lats, lons = XYZ2LatLon(tracers[:, :3])
        ages = tracers[:, 3]

        return lons, lats, ages
