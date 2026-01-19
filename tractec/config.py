"""Configuration classes for tractec package."""

from dataclasses import dataclass
import numpy as np


@dataclass
class TracerConfig:
    """
    Configuration parameters for seafloor age tracking.

    Parameters match GPlately's SeafloorGrid for compatibility.

    Attributes
    ----------
    time_step : float
        Time step size in Myr (default: 1.0)
    earth_radius : float
        Earth's radius in meters (default: 6.3781e6)

    Collision Detection (C++ backend)
    ----------------------------------
    velocity_delta_threshold : float
        Minimum velocity difference to trigger collision check, in km/Myr.
        Converted to cm/yr (divide by 10) for pygplates C++ API.
        Default: 7.0 km/Myr (= 0.7 cm/yr)
    distance_threshold_per_myr : float
        Base distance threshold for collision detection, in km/Myr.
        Default: 10.0 km/Myr

    Initialization
    --------------
    default_refinement_levels : int
        Number of icosahedral mesh refinement levels.
        Level 5 = ~10,242 points, Level 6 = ~40,962 points.
        Default: 5
    initial_ocean_mean_spreading_rate : float
        Mean spreading rate in mm/yr for initial age calculation.
        Used to compute age = distance / (rate / 2).
        Default: 75.0 mm/yr (GPlately default)

    MOR Seed Generation
    -------------------
    ridge_sampling_degrees : float
        Ridge tessellation resolution in degrees (~50 km at equator).
        Default: 0.5 degrees
    spreading_offset_degrees : float
        Angle to rotate new points off ridge, in degrees.
        Default: 0.01 degrees (~1 km)

    Continental Handling
    --------------------
    continental_cache_size : int
        Number of timesteps to cache for continental polygon queries.
        Default: 10

    Reinitialization
    ----------------
    reinit_k_neighbors : int
        Number of nearest neighbors for interpolation during reinitialization.
        Default: 5
    reinit_max_distance : float
        Maximum distance (meters) for valid interpolation neighbors.
        Default: 500e3 (500 km)

    Examples
    --------
    >>> # Use default configuration (GPlately-compatible)
    >>> config = TracerConfig()
    >>>
    >>> # Custom configuration with higher resolution
    >>> config = TracerConfig(
    ...     default_refinement_levels=6,  # ~40,962 points
    ...     ridge_sampling_degrees=0.25,  # ~25 km resolution
    ...     time_step=0.5                 # 0.5 Myr timesteps
    ... )
    """

    # Time stepping
    time_step: float = 1.0  # Myr
    earth_radius: float = 6.3781e6  # meters

    # Collision detection (C++ backend - GPlately compatible)
    # These are passed to pygplates.ReconstructedGeometryTimeSpan.DefaultDeactivatePoints
    velocity_delta_threshold: float = 7.0  # km/Myr (converted to 0.7 cm/yr for API)
    distance_threshold_per_myr: float = 10.0  # km/Myr

    # Initialization - icosahedral mesh
    default_refinement_levels: int = 5  # ~10,242 points
    initial_ocean_mean_spreading_rate: float = 75.0  # mm/yr (GPlately default)

    # MOR seed generation
    ridge_sampling_degrees: float = 0.5  # ~50 km at equator
    spreading_offset_degrees: float = 0.01  # ~1 km offset from ridge

    # Continental polygon caching
    continental_cache_size: int = 10

    # Reinitialization parameters
    reinit_k_neighbors: int = 5
    reinit_max_distance: float = 500e3  # meters

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.time_step <= 0:
            raise ValueError(f"time_step must be positive, got {self.time_step}")
        if self.earth_radius <= 0:
            raise ValueError(f"earth_radius must be positive, got {self.earth_radius}")
        if self.velocity_delta_threshold < 0:
            raise ValueError(
                f"velocity_delta_threshold must be non-negative, "
                f"got {self.velocity_delta_threshold}"
            )
        if self.distance_threshold_per_myr < 0:
            raise ValueError(
                f"distance_threshold_per_myr must be non-negative, "
                f"got {self.distance_threshold_per_myr}"
            )
        if self.default_refinement_levels < 0:
            raise ValueError(
                f"default_refinement_levels must be non-negative, "
                f"got {self.default_refinement_levels}"
            )
        if self.initial_ocean_mean_spreading_rate <= 0:
            raise ValueError(
                f"initial_ocean_mean_spreading_rate must be positive, "
                f"got {self.initial_ocean_mean_spreading_rate}"
            )
        if self.ridge_sampling_degrees <= 0:
            raise ValueError(
                f"ridge_sampling_degrees must be positive, "
                f"got {self.ridge_sampling_degrees}"
            )
        if self.spreading_offset_degrees <= 0:
            raise ValueError(
                f"spreading_offset_degrees must be positive, "
                f"got {self.spreading_offset_degrees}"
            )
        if self.continental_cache_size < 0:
            raise ValueError(
                f"continental_cache_size must be non-negative, "
                f"got {self.continental_cache_size}"
            )
        if self.reinit_k_neighbors < 1:
            raise ValueError(
                f"reinit_k_neighbors must be at least 1, "
                f"got {self.reinit_k_neighbors}"
            )
        if self.reinit_max_distance <= 0:
            raise ValueError(
                f"reinit_max_distance must be positive, "
                f"got {self.reinit_max_distance}"
            )

    @property
    def velocity_delta_threshold_cm_yr(self) -> float:
        """
        Velocity threshold in cm/yr for pygplates C++ API.

        The C++ API expects cm/yr, while we store km/Myr for readability.
        Conversion: 1 km/Myr = 0.1 cm/yr
        """
        return self.velocity_delta_threshold / 10.0

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            'time_step': self.time_step,
            'earth_radius': self.earth_radius,
            'velocity_delta_threshold': self.velocity_delta_threshold,
            'distance_threshold_per_myr': self.distance_threshold_per_myr,
            'default_refinement_levels': self.default_refinement_levels,
            'initial_ocean_mean_spreading_rate': self.initial_ocean_mean_spreading_rate,
            'ridge_sampling_degrees': self.ridge_sampling_degrees,
            'spreading_offset_degrees': self.spreading_offset_degrees,
            'continental_cache_size': self.continental_cache_size,
            'reinit_k_neighbors': self.reinit_k_neighbors,
            'reinit_max_distance': self.reinit_max_distance,
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with configuration parameters

        Returns
        -------
        TracerConfig
            Configuration object
        """
        return cls(**config_dict)
