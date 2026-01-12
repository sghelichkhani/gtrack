"""Tests for core functionality."""

import numpy as np
import pytest
from tractec import TracerConfig
from tractec.geometry import LatLon2XYZ, XYZ2LatLon


class TestTracerConfig:
    """Test TracerConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = TracerConfig()

        assert config.ridge_resolution == 50e3
        assert config.subduction_resolution == 20e3
        assert config.ridge_offset == 1e3
        assert config.continental_tolerance == 150e3
        assert config.subduction_tolerance == 100e3
        assert config.time_step == 1.0
        assert config.earth_radius == 6.3781e6

    def test_custom_config(self):
        """Test custom configuration."""
        config = TracerConfig(
            ridge_resolution=25e3,
            time_step=0.5
        )

        assert config.ridge_resolution == 25e3
        assert config.time_step == 0.5
        # Other values should be defaults
        assert config.subduction_resolution == 20e3

    def test_config_validation(self):
        """Test configuration validation."""
        # Negative ridge_resolution should raise error
        with pytest.raises(ValueError, match="ridge_resolution"):
            TracerConfig(ridge_resolution=-1)

        # Negative time_step should raise error
        with pytest.raises(ValueError, match="time_step"):
            TracerConfig(time_step=-1)

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        config = TracerConfig(ridge_resolution=25e3)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['ridge_resolution'] == 25e3
        assert 'time_step' in config_dict

    def test_config_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'ridge_resolution': 25e3,
            'time_step': 0.5,
        }
        config = TracerConfig.from_dict(config_dict)

        assert config.ridge_resolution == 25e3
        assert config.time_step == 0.5


class TestOptimizations:
    """Test that optimized functions produce reasonable results."""

    def test_ridge_tracer_generation(self):
        """Test vectorized ridge tracer generation."""
        from tractec.ridges import add_ridge_tracers_vectorized

        # Create simple ridge points on equator with appropriate spacing
        # Points should be spaced < res_ridge * 1.5 apart
        lats = np.zeros(20)
        lons = np.linspace(0, 5, 20)  # 5 degrees span with close points
        latlon = np.column_stack([lats, lons])
        ridge_points = LatLon2XYZ(latlon)

        # Generate tracers with appropriate resolution
        tracers = add_ridge_tracers_vectorized(
            ridge_points,
            epsilon_R=1e3,
            res_ridge=100e3  # 100 km resolution
        )

        # Should produce tracers
        assert len(tracers) > 0
        # Should have 4 columns (x, y, z, age)
        assert tracers.shape[1] == 4
        # Age should be 0
        assert np.all(tracers[:, 3] == 0)

    def test_subduction_check_empty(self):
        """Test subduction check with no plate changes."""
        from tractec.subduction import check_for_subduction_optimized

        # Create dummy tracers
        tracers = np.random.rand(10, 4)
        plate_ids = np.ones(10, dtype=int)  # All same plate
        subduction_points = np.random.rand(5, 3)

        # No plate changes, so no subduction
        result = check_for_subduction_optimized(
            tracers, plate_ids, plate_ids, subduction_points
        )

        # All tracers should survive
        assert len(result) == len(tracers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
