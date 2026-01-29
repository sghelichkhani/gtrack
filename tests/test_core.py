"""Tests for core functionality."""

import numpy as np
import pytest
from gtrack import TracerConfig
from gtrack.geometry import LatLon2XYZ, XYZ2LatLon


class TestTracerConfig:
    """Test TracerConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = TracerConfig()

        assert config.time_step == 1.0
        assert config.earth_radius == 6.3781e6
        assert config.velocity_delta_threshold == 7.0
        assert config.distance_threshold_per_myr == 10.0
        assert config.default_refinement_levels == 5
        assert config.initial_ocean_mean_spreading_rate == 75.0
        assert config.ridge_sampling_degrees == 0.5
        assert config.spreading_offset_degrees == 0.01

    def test_custom_config(self):
        """Test custom configuration."""
        config = TracerConfig(
            time_step=0.5,
            default_refinement_levels=6,
            ridge_sampling_degrees=0.25
        )

        assert config.time_step == 0.5
        assert config.default_refinement_levels == 6
        assert config.ridge_sampling_degrees == 0.25
        # Other values should be defaults
        assert config.velocity_delta_threshold == 7.0

    def test_config_validation(self):
        """Test configuration validation."""
        # Negative time_step should raise error
        with pytest.raises(ValueError, match="time_step"):
            TracerConfig(time_step=-1)

        # Negative refinement levels should raise error
        with pytest.raises(ValueError, match="default_refinement_levels"):
            TracerConfig(default_refinement_levels=-1)

        # Non-positive spreading rate should raise error
        with pytest.raises(ValueError, match="initial_ocean_mean_spreading_rate"):
            TracerConfig(initial_ocean_mean_spreading_rate=0)

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        config = TracerConfig(time_step=0.5)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['time_step'] == 0.5
        assert 'velocity_delta_threshold' in config_dict
        assert 'default_refinement_levels' in config_dict

    def test_config_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'time_step': 0.5,
            'default_refinement_levels': 6,
        }
        config = TracerConfig.from_dict(config_dict)

        assert config.time_step == 0.5
        assert config.default_refinement_levels == 6

    def test_velocity_threshold_conversion(self):
        """Test velocity threshold unit conversion property."""
        config = TracerConfig(velocity_delta_threshold=7.0)

        # 7.0 km/Myr should convert to 0.7 cm/yr
        assert config.velocity_delta_threshold_cm_yr == 0.7


class TestMeshGeneration:
    """Test icosahedral mesh generation."""

    def test_create_icosahedral_mesh_latlon(self):
        """Test icosahedral mesh creation returning lat/lon."""
        from gtrack.mesh import create_icosahedral_mesh_latlon, mesh_point_count

        lats, lons = create_icosahedral_mesh_latlon(refinement_levels=3)

        # Check shapes match
        assert len(lats) == len(lons)

        # Check we get the expected number of points (10 * 4^level + 2)
        expected = mesh_point_count(3)  # 642 points
        assert len(lats) == expected

        # Check latitude range
        assert np.all(lats >= -90)
        assert np.all(lats <= 90)

        # Check longitude range
        assert np.all(lons >= -180)
        assert np.all(lons <= 180)

    def test_create_icosahedral_mesh_xyz(self):
        """Test icosahedral mesh creation returning XYZ."""
        from gtrack.mesh import create_icosahedral_mesh_xyz

        xyz = create_icosahedral_mesh_xyz(refinement_levels=3)

        # Check shape
        assert xyz.shape[1] == 3

        # Check points are on unit sphere (default radius=1)
        radii = np.linalg.norm(xyz, axis=1)
        np.testing.assert_allclose(radii, 1.0, rtol=1e-10)

    def test_mesh_point_count(self):
        """Test mesh point count formula provides an estimate."""
        from gtrack.mesh import mesh_point_count

        # The formula 10 * 4^level + 2 gives an approximate count
        # Level 0 should give exactly 12 (original icosahedron)
        assert mesh_point_count(0) == 12

        # Higher levels grow exponentially
        assert mesh_point_count(1) > mesh_point_count(0)
        assert mesh_point_count(5) > 10000


class TestMORSeeds:
    """Test MOR seed point generation functions."""

    def test_get_stage_rotation_returns_none_without_plates(self):
        """Test that get_stage_rotation returns None for features without plate IDs."""
        from gtrack.mor_seeds import get_stage_rotation_for_reconstructed_geometry
        import pygplates

        # Create a mock feature without plate IDs
        feature = pygplates.Feature()
        feature.set_geometry(pygplates.PointOnSphere(0, 0))

        # Create a simple rotation model (won't be used since we lack plate IDs)
        rotation_model = pygplates.RotationModel([])

        result = get_stage_rotation_for_reconstructed_geometry(
            feature, rotation_model, spreading_time=100.0
        )

        assert result is None


class TestInitialConditions:
    """Test initial age calculation functions."""

    def test_default_age_distance_law(self):
        """Test the default age-distance formula."""
        from gtrack.initial_conditions import default_age_distance_law

        distances_km = np.array([75.0, 150.0, 300.0])
        spreading_rate_mm_yr = 75.0  # = 75 km/Myr

        ages = default_age_distance_law(distances_km, spreading_rate_mm_yr)

        # age = distance / (rate / 2)
        # For 75 km at 75 mm/yr half-rate (37.5 km/Myr): 75 / 37.5 = 2 Myr
        expected = np.array([2.0, 4.0, 8.0])
        np.testing.assert_allclose(ages, expected)


class TestBoundaries:
    """Test boundary caching and extraction."""

    def test_continental_polygon_cache_empty(self):
        """Test continental cache behavior with no polygons."""
        from gtrack.boundaries import ContinentalPolygonCache
        import pygplates

        # Create empty feature collection
        empty_features = pygplates.FeatureCollection()
        rotation_model = pygplates.RotationModel([])

        cache = ContinentalPolygonCache(empty_features, rotation_model)

        # Query should return empty polygons
        polygons = cache.get_polygons(100.0)
        assert len(polygons) == 0

        # Mask should be all False (no continental points)
        lats = np.array([0.0, 45.0, -45.0])
        lons = np.array([0.0, 90.0, -90.0])
        mask = cache.get_continental_mask(lats, lons, 100.0)

        assert mask.shape == (3,)
        assert not mask.any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
