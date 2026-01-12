"""Configuration classes for TracTec package."""

from dataclasses import dataclass, field


@dataclass
class TracerConfig:
    """
    Configuration parameters for tracer computation.

    Attributes
    ----------
    ridge_resolution : float
        Sampling resolution for mid-ocean ridges in meters (default: 50 km)
    subduction_resolution : float
        Sampling resolution for subduction zones in meters (default: 20 km)
    ridge_offset : float
        Distance from ridge to add new tracers in meters (default: 1 km)
    continental_tolerance : float
        Proximity tolerance for continental interaction in meters (default: 150 km)
    subduction_tolerance : float
        Proximity tolerance for subduction detection in meters (default: 100 km)
    time_step : float
        Time step size in Myr (default: 1.0)
    earth_radius : float
        Earth's radius in meters (default: 6.3781e6)

    Examples
    --------
    >>> # Use default configuration
    >>> config = TracerConfig()
    >>>
    >>> # Custom configuration with higher resolution
    >>> config = TracerConfig(
    ...     ridge_resolution=25e3,  # 25 km instead of 50 km
    ...     time_step=0.5           # 0.5 Myr instead of 1.0 Myr
    ... )
    """

    ridge_resolution: float = 50e3  # meters
    subduction_resolution: float = 20e3  # meters
    ridge_offset: float = 1e3  # meters (epsilon_R in original)
    continental_tolerance: float = 150e3  # meters
    subduction_tolerance: float = 100e3  # meters
    time_step: float = 1.0  # Myr
    earth_radius: float = 6.3781e6  # meters

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.ridge_resolution <= 0:
            raise ValueError(f"ridge_resolution must be positive, got {self.ridge_resolution}")
        if self.subduction_resolution <= 0:
            raise ValueError(f"subduction_resolution must be positive, got {self.subduction_resolution}")
        if self.ridge_offset < 0:
            raise ValueError(f"ridge_offset must be non-negative, got {self.ridge_offset}")
        if self.continental_tolerance < 0:
            raise ValueError(f"continental_tolerance must be non-negative, got {self.continental_tolerance}")
        if self.subduction_tolerance < 0:
            raise ValueError(f"subduction_tolerance must be non-negative, got {self.subduction_tolerance}")
        if self.time_step <= 0:
            raise ValueError(f"time_step must be positive, got {self.time_step}")
        if self.earth_radius <= 0:
            raise ValueError(f"earth_radius must be positive, got {self.earth_radius}")

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            'ridge_resolution': self.ridge_resolution,
            'subduction_resolution': self.subduction_resolution,
            'ridge_offset': self.ridge_offset,
            'continental_tolerance': self.continental_tolerance,
            'subduction_tolerance': self.subduction_tolerance,
            'time_step': self.time_step,
            'earth_radius': self.earth_radius,
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
