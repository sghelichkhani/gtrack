"""
Seafloor age tracking using Lagrangian particle tracking.

This module provides the main API for computing seafloor ages
as geological age decreases toward present (0 Ma).
"""

import numpy as np
from typing import Dict, List, Optional, Union
from .model import SeafloorAgeModel
from .config import TracerConfig
from .geometry import XYZ2LatLon
from .point_rotation import PointCloud


class SeafloorAgeTracker:
    """
    Seafloor age tracker using Lagrangian particle tracking.

    Maintains tracer state and provides incremental evolution
    as geological age decreases toward present (0 Ma).

    Key features:
    - Stateful: Maintains tracers in memory between updates
    - Incremental: Only evolves from current state to new state
    - PointCloud output: Returns PointCloud with 'age' property
    - Checkpointing: Save/restore state for restarts

    Parameters
    ----------
    rotation_files : list of str
        Paths to rotation model files (.rot).
    topology_files : list of str
        Paths to topology/plate boundary files (.gpmlz).
    continental_polygons : str
        Path to continental polygon file (.gpmlz).
    config : TracerConfig, optional
        Configuration parameters. If None, uses defaults.
    preload_boundaries : bool, default=True
        Whether to pre-load all boundaries. Set False for memory-constrained systems.
    verbose : bool, default=True
        Print progress information.

    Examples
    --------
    >>> # Initialize tracker
    >>> tracker = SeafloorAgeTracker(
    ...     rotation_files=['rotations.rot'],
    ...     topology_files=['topologies.gpmlz'],
    ...     continental_polygons='continents.gpmlz'
    ... )
    >>>
    >>> # Initialize tracers at ridges for 200 Ma
    >>> tracker.initialize(starting_age=200)
    >>>
    >>> # Step forward (decreasing geological age toward present)
    >>> for target_age in range(199, -1, -1):
    ...     cloud = tracker.step_to(target_age)
    ...     xyz = cloud.xyz
    ...     ages = cloud.get_property('age')
    """

    def __init__(
        self,
        rotation_files: Union[str, List[str]],
        topology_files: Union[str, List[str]],
        continental_polygons: str,
        config: Optional[TracerConfig] = None,
        preload_boundaries: bool = True,
        verbose: bool = True
    ):
        self.verbose = verbose

        # Ensure lists
        if isinstance(rotation_files, str):
            rotation_files = [rotation_files]
        if isinstance(topology_files, str):
            topology_files = [topology_files]

        self._rotation_files = rotation_files
        self._topology_files = topology_files
        self._continental_polygons = continental_polygons
        self._config = config if config else TracerConfig()

        if self.verbose:
            print("Initializing SeafloorAgeTracker...")

        # Initialize TracTec model
        self._model = SeafloorAgeModel(
            rotation_files=rotation_files,
            topology_files=topology_files,
            continental_polygons=continental_polygons,
            config=self._config
        )

        self._preload_boundaries = preload_boundaries

        # State variables
        self._current_age: Optional[float] = None
        self._tracers: Optional[np.ndarray] = None
        self._initialized = False

        if self.verbose:
            print("  Initialization complete.")

    def initialize(self, starting_age: float) -> int:
        """
        Initialize tracers at ridges for given geological age.

        Parameters
        ----------
        starting_age : float
            Starting geological age (Ma). Tracers are placed at
            ridge locations at this age.

        Returns
        -------
        int
            Number of tracers initialized.

        Examples
        --------
        >>> tracker.initialize(starting_age=200)
        Initialized with 12534 tracers at 200 Ma
        12534
        """
        if self.verbose:
            print(f"\nInitializing tracers at {starting_age} Ma...")

        # Pre-load boundaries if requested
        if self._preload_boundaries:
            if self.verbose:
                print("  Pre-loading boundaries...")
            self._model.preload_boundaries(range(0, int(starting_age) + 1))

            if self.verbose:
                memory_info = self._model._boundary_cache.get_memory_usage()
                print(f"  Cached {memory_info['num_timesteps_cached']} timesteps")
                print(f"  Memory: {memory_info['total_mb']:.1f} MB")

        self._tracers = self._model._initialize_tracers(starting_age)
        self._current_age = starting_age
        self._initialized = True

        num_tracers = len(self._tracers)

        if self.verbose:
            print(f"  Initialized with {num_tracers} tracers at {starting_age} Ma")

        return num_tracers

    def initialize_from_cloud(
        self,
        cloud: PointCloud,
        current_age: float
    ) -> int:
        """
        Initialize from existing PointCloud.

        Use this to restart from a checkpoint or to provide
        custom initial tracer positions.

        Parameters
        ----------
        cloud : PointCloud
            Point cloud with 'age' property containing material
            age of each tracer (time since ridge formation).
        current_age : float
            Current geological age (Ma).

        Returns
        -------
        int
            Number of tracers initialized.

        Raises
        ------
        ValueError
            If cloud does not have 'age' property.

        Examples
        --------
        >>> # Restart from checkpoint
        >>> cloud, metadata = checkpoint.load('state.npz')
        >>> tracker.initialize_from_cloud(cloud, metadata['geological_age'])
        """
        if 'age' not in cloud.properties:
            raise ValueError(
                "PointCloud must have 'age' property. "
                "This should contain the material age of each tracer."
            )

        if self.verbose:
            print(f"\nInitializing from PointCloud at {current_age} Ma...")

        # Pre-load boundaries if requested
        if self._preload_boundaries:
            if self.verbose:
                print("  Pre-loading boundaries...")
            self._model.preload_boundaries(range(0, int(current_age) + 1))

        # Convert PointCloud to internal tracer format: (x, y, z, age)
        ages = cloud.get_property('age')
        self._tracers = np.column_stack([cloud.xyz, ages])
        self._current_age = current_age
        self._initialized = True

        if self.verbose:
            print(f"  Initialized with {len(self._tracers)} tracers at {current_age} Ma")

        return len(self._tracers)

    def step_to(self, target_age: float) -> PointCloud:
        """
        Evolve tracers to target geological age.

        Can only step forward (decreasing geological age toward 0).

        Parameters
        ----------
        target_age : float
            Target geological age (Ma). Must be less than current_age.

        Returns
        -------
        PointCloud
            Point cloud with 'age' property containing material ages.

        Raises
        ------
        RuntimeError
            If tracker is not initialized.
        ValueError
            If target_age > current_age (can only go forward).

        Examples
        --------
        >>> cloud = tracker.step_to(150)
        >>> xyz = cloud.xyz
        >>> ages = cloud.get_property('age')
        """
        if not self._initialized:
            raise RuntimeError(
                "Must call initialize() or initialize_from_cloud() first!"
            )

        if target_age > self._current_age:
            raise ValueError(
                f"Can only step forward (decreasing age). "
                f"Current age: {self._current_age}, target: {target_age}"
            )

        if target_age == self._current_age:
            if self.verbose:
                print(f"Already at {target_age} Ma, no update needed")
            return self.get_current_state()

        if self.verbose:
            print(f"\nEvolving: {self._current_age} Ma -> {target_age} Ma")

        # Evolve tracers
        self._tracers = self._model._evolve_tracers(
            self._tracers,
            from_time=self._current_age,
            to_time=target_age
        )

        self._current_age = target_age

        if self.verbose:
            print(f"  Current tracer count: {len(self._tracers)}")

        return self.get_current_state()

    def get_current_state(self) -> PointCloud:
        """
        Get current tracers as PointCloud without evolving.

        Returns
        -------
        PointCloud
            Current tracer positions with 'age' property.
        """
        if self._tracers is None or len(self._tracers) == 0:
            # Return empty PointCloud
            return PointCloud(
                xyz=np.zeros((0, 3)),
                properties={'age': np.array([])}
            )

        return PointCloud(
            xyz=self._tracers[:, :3].copy(),
            properties={'age': self._tracers[:, 3].copy()}
        )

    @property
    def current_age(self) -> Optional[float]:
        """Current geological age."""
        return self._current_age

    @property
    def n_tracers(self) -> int:
        """Number of tracers."""
        return len(self._tracers) if self._tracers is not None else 0

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save state to checkpoint file.

        Parameters
        ----------
        filepath : str
            Path to save checkpoint (.npz format).

        Examples
        --------
        >>> tracker.save_checkpoint('checkpoint_150Ma.npz')
        """
        from .io_formats import PointCloudCheckpoint

        if not self._initialized:
            raise RuntimeError("No state to save - not initialized")

        cloud = self.get_current_state()
        checkpoint = PointCloudCheckpoint()
        checkpoint.save(
            cloud,
            filepath,
            geological_age=self._current_age,
            metadata={
                'time_step': self._config.time_step,
            }
        )

        if self.verbose:
            print(f"Saved checkpoint to {filepath} ({self._current_age} Ma)")

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load state from checkpoint file.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file.

        Examples
        --------
        >>> tracker.load_checkpoint('checkpoint_150Ma.npz')
        """
        from .io_formats import PointCloudCheckpoint

        checkpoint = PointCloudCheckpoint()
        cloud, metadata = checkpoint.load(filepath)

        geological_age = metadata.get('geological_age')
        if geological_age is None:
            raise ValueError("Checkpoint missing 'geological_age' in metadata")

        self.initialize_from_cloud(cloud, geological_age)

        if self.verbose:
            print(f"Loaded checkpoint from {filepath}")
            print(f"  Geological age: {self._current_age} Ma")
            print(f"  Tracers: {len(self._tracers)}")

    def get_statistics(self) -> Dict:
        """
        Get statistics about current tracer state.

        Returns
        -------
        dict
            Statistics including mean age, max age, coverage, etc.
        """
        if not self._initialized or len(self._tracers) == 0:
            return {
                'count': 0,
                'mean_age': 0,
                'max_age': 0,
                'min_age': 0,
                'geological_age': self._current_age
            }

        ages = self._tracers[:, 3]

        return {
            'count': len(ages),
            'mean_age': float(np.mean(ages)),
            'max_age': float(np.max(ages)),
            'min_age': float(np.min(ages)),
            'std_age': float(np.std(ages)),
            'geological_age': self._current_age
        }

    @classmethod
    def compute_ages(
        cls,
        target_age: float,
        starting_age: float,
        rotation_files: Union[str, List[str]],
        topology_files: Union[str, List[str]],
        continental_polygons: str,
        config: Optional[TracerConfig] = None,
        verbose: bool = True
    ) -> PointCloud:
        """
        One-shot computation of seafloor ages (functional interface).

        Creates a tracker, initializes at starting_age, and evolves
        to target_age in a single call. Use this when you don't need
        incremental updates.

        Parameters
        ----------
        target_age : float
            Target geological age (Ma).
        starting_age : float
            Starting geological age (Ma).
        rotation_files : list of str
            Paths to rotation model files (.rot).
        topology_files : list of str
            Paths to topology/plate boundary files (.gpmlz).
        continental_polygons : str
            Path to continental polygon file (.gpmlz).
        config : TracerConfig, optional
            Configuration parameters.
        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        PointCloud
            Point cloud with 'age' property.

        Examples
        --------
        >>> cloud = SeafloorAgeTracker.compute_ages(
        ...     target_age=100,
        ...     starting_age=200,
        ...     rotation_files=['rotations.rot'],
        ...     topology_files=['topologies.gpmlz'],
        ...     continental_polygons='continents.gpmlz'
        ... )
        >>> ages = cloud.get_property('age')
        """
        tracker = cls(
            rotation_files=rotation_files,
            topology_files=topology_files,
            continental_polygons=continental_polygons,
            config=config,
            preload_boundaries=True,
            verbose=verbose
        )

        tracker.initialize(starting_age)
        return tracker.step_to(target_age)


# Backwards compatibility aliases (deprecated)
HPCSeafloorAgeTracker = SeafloorAgeTracker
MemoryEfficientSeafloorAgeTracker = SeafloorAgeTracker
