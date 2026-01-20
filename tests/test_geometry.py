"""Tests for geometry module."""

import numpy as np
import pytest
from gtrack.geometry import (
    LatLon2XYZ,
    XYZ2LatLon,
    RefineGreatCircleArcSegment,
    Segments2Points,
    normalize_to_sphere,
    inverse_distance_weighted_interpolation,
    compute_mesh_spacing_km,
    EARTH_RADIUS,
)


class TestCoordinateTransformations:
    """Test coordinate transformation functions."""

    def test_latlon_to_xyz_single_point(self):
        """Test conversion of a single lat/lon point to XYZ."""
        # Point at equator and prime meridian
        latlon = np.array([[0.0, 0.0]])
        xyz = LatLon2XYZ(latlon)

        assert xyz.shape == (1, 3)
        # Should be at (R, 0, 0)
        np.testing.assert_allclose(xyz[0], [EARTH_RADIUS, 0, 0], atol=1e-6)

    def test_latlon_to_xyz_north_pole(self):
        """Test conversion of north pole to XYZ."""
        latlon = np.array([[90.0, 0.0]])
        xyz = LatLon2XYZ(latlon)

        # Should be at (0, 0, R)
        np.testing.assert_allclose(xyz[0], [0, 0, EARTH_RADIUS], atol=1e-6)

    def test_latlon_to_xyz_south_pole(self):
        """Test conversion of south pole to XYZ."""
        latlon = np.array([[-90.0, 0.0]])
        xyz = LatLon2XYZ(latlon)

        # Should be at (0, 0, -R)
        np.testing.assert_allclose(xyz[0], [0, 0, -EARTH_RADIUS], atol=1e-6)

    def test_latlon_to_xyz_multiple_points(self):
        """Test conversion of multiple points."""
        latlon = np.array([
            [0.0, 0.0],
            [0.0, 90.0],
            [0.0, 180.0],
            [0.0, -90.0],
        ])
        xyz = LatLon2XYZ(latlon)

        assert xyz.shape == (4, 3)
        # All points should be on the equator (z=0) and at distance R
        np.testing.assert_allclose(xyz[:, 2], 0, atol=1e-6)
        distances = np.linalg.norm(xyz, axis=1)
        np.testing.assert_allclose(distances, EARTH_RADIUS, atol=1e-6)

    def test_xyz_to_latlon_single_point(self):
        """Test conversion of single XYZ point to lat/lon."""
        xyz = np.array([[EARTH_RADIUS, 0, 0]])
        lats, lons = XYZ2LatLon(xyz)

        assert lats.shape == (1,)
        assert lons.shape == (1,)
        np.testing.assert_allclose(lats[0], 0, atol=1e-6)
        np.testing.assert_allclose(lons[0], 0, atol=1e-6)

    def test_xyz_to_latlon_poles(self):
        """Test conversion of poles."""
        xyz = np.array([
            [0, 0, EARTH_RADIUS],   # North pole
            [0, 0, -EARTH_RADIUS],  # South pole
        ])
        lats, lons = XYZ2LatLon(xyz)

        np.testing.assert_allclose(lats, [90, -90], atol=1e-6)

    def test_coordinate_transform_round_trip(self):
        """Test that LatLon -> XYZ -> LatLon preserves coordinates."""
        original_latlon = np.array([
            [0.0, 0.0],
            [45.0, 90.0],
            [-30.0, -120.0],
            [60.0, 45.0],
        ])

        xyz = LatLon2XYZ(original_latlon)
        lats, lons = XYZ2LatLon(xyz)
        recovered_latlon = np.column_stack([lats, lons])

        np.testing.assert_allclose(recovered_latlon, original_latlon, atol=1e-9)

    def test_xyz_points_on_sphere(self):
        """Test that LatLon2XYZ produces points on the sphere."""
        latlon = np.random.rand(100, 2) * np.array([[180, 360]]) - np.array([[90, 180]])
        xyz = LatLon2XYZ(latlon)

        distances = np.linalg.norm(xyz, axis=1)
        np.testing.assert_allclose(distances, EARTH_RADIUS, rtol=1e-10)


class TestGreatCircleRefinement:
    """Test great circle arc refinement."""

    def test_refine_same_point(self):
        """Test refinement when start and end points are the same."""
        p1 = (0.0, 0.0)
        p2 = (0.0, 0.0)
        lats, lons = RefineGreatCircleArcSegment(p1, p2, 5)

        assert len(lats) == 6  # N+1 points
        assert len(lons) == 6
        # All points should be the same
        np.testing.assert_allclose(lats, 0, atol=1e-6)
        np.testing.assert_allclose(lons, 0, atol=1e-6)

    def test_refine_equatorial_segment(self):
        """Test refinement of segment along equator."""
        p1 = (0.0, 0.0)
        p2 = (0.0, 90.0)
        lats, lons = RefineGreatCircleArcSegment(p1, p2, 2)

        assert len(lats) == 3
        assert len(lons) == 3
        # All latitudes should be close to 0 (on equator)
        np.testing.assert_allclose(lats, 0, atol=1)
        # Longitudes should progress from 0 to 90
        assert lons[0] < lons[1] < lons[2]

    def test_refine_meridional_segment(self):
        """Test refinement of segment along a meridian."""
        p1 = (0.0, 0.0)
        p2 = (90.0, 0.0)
        lats, lons = RefineGreatCircleArcSegment(p1, p2, 2)

        assert len(lats) == 3
        # Latitudes should progress from 0 to 90
        assert lats[0] < lats[1] < lats[2]

    def test_refined_points_on_sphere(self):
        """Test that refined points lie on the sphere."""
        p1 = (45.0, 30.0)
        p2 = (-30.0, 120.0)
        lats, lons = RefineGreatCircleArcSegment(p1, p2, 10)

        # Convert to XYZ and check distances
        latlon = np.column_stack([lats, lons])
        xyz = LatLon2XYZ(latlon)
        distances = np.linalg.norm(xyz, axis=1)
        np.testing.assert_allclose(distances, EARTH_RADIUS, rtol=1e-10)


class TestSegments2Points:
    """Test segment-to-points conversion."""

    def test_single_segment(self):
        """Test conversion of a single segment."""
        # Segment from (0,0) to (0,10) degrees
        segments = np.array([[0, 0, 0, 10]])
        lats, lons = Segments2Points(segments, 100e3)  # 100 km resolution

        # Should produce multiple points
        assert len(lats) > 2
        assert len(lons) > 2

    def test_multiple_segments(self):
        """Test conversion of multiple segments."""
        segments = np.array([
            [0, 0, 0, 10],
            [0, 10, 0, 20],
            [0, 20, 0, 30],
        ])
        lats, lons = Segments2Points(segments, 100e3)

        # Should produce more points than segments
        assert len(lats) > len(segments)

    def test_zero_length_segment(self):
        """Test that zero-length segments are handled."""
        segments = np.array([
            [0, 0, 0, 0],  # Zero length
            [0, 0, 0, 10], # Non-zero length
        ])
        lats, lons = Segments2Points(segments, 100e3)

        # Should still work, just skip the zero-length segment
        assert len(lats) >= 0


class TestNormalization:
    """Test normalization to sphere."""

    def test_normalize_unit_vectors(self):
        """Test normalizing unit vectors."""
        xyz = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        normalized = normalize_to_sphere(xyz, radius=2.0)

        distances = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(distances, 2.0, rtol=1e-10)

    def test_normalize_arbitrary_points(self):
        """Test normalizing arbitrary points."""
        xyz = np.random.randn(20, 3) * 1000
        normalized = normalize_to_sphere(xyz, radius=EARTH_RADIUS)

        distances = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(distances, EARTH_RADIUS, rtol=1e-10)

    def test_normalize_default_radius(self):
        """Test normalization with default Earth radius."""
        xyz = np.array([[100, 200, 300]])
        normalized = normalize_to_sphere(xyz)

        distance = np.linalg.norm(normalized[0])
        np.testing.assert_allclose(distance, EARTH_RADIUS, rtol=1e-10)


class TestIDWInterpolation:
    """Test inverse distance weighted interpolation."""

    def test_idw_basic(self):
        """Test basic IDW interpolation."""
        # Two neighbors with values 10 and 20 at distances 1 and 2
        values = np.array([[10, 20]])
        distances = np.array([[1.0, 2.0]])
        result = inverse_distance_weighted_interpolation(values, distances)

        # Expected: (10/1 + 20/2) / (1/1 + 1/2) = 20 / 1.5 = 13.33
        expected = 20.0 / 1.5
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_idw_equal_distances(self):
        """Test IDW with equal distances (simple average)."""
        values = np.array([[10, 20, 30]])
        distances = np.array([[1.0, 1.0, 1.0]])
        result = inverse_distance_weighted_interpolation(values, distances)

        # Equal distances means simple average
        expected = 20.0
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_idw_zero_distance(self):
        """Test IDW with zero distance (exact match)."""
        values = np.array([[10, 20, 30]])
        distances = np.array([[0.0, 1.0, 2.0]])
        result = inverse_distance_weighted_interpolation(values, distances)

        # Zero distance means use that value directly
        expected = 10.0
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_idw_single_neighbor(self):
        """Test IDW with single neighbor (k=1)."""
        values = np.array([[42]])
        distances = np.array([[5.0]])
        result = inverse_distance_weighted_interpolation(values, distances)

        expected = 42.0
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_idw_multiple_points(self):
        """Test IDW with multiple query points."""
        values = np.array([
            [10, 20],
            [30, 40],
            [50, 60],
        ])
        distances = np.array([
            [1.0, 2.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ])
        result = inverse_distance_weighted_interpolation(values, distances)

        assert result.shape == (3,)
        # Point 1: (10/1 + 20/2) / (1 + 0.5) = 20/1.5 = 13.33
        # Point 2: (30 + 40) / 2 = 35
        # Point 3: (50/2 + 60/1) / (0.5 + 1) = 85/1.5 = 56.67
        np.testing.assert_allclose(result[0], 20.0 / 1.5, rtol=1e-10)
        np.testing.assert_allclose(result[1], 35.0, rtol=1e-10)
        np.testing.assert_allclose(result[2], 85.0 / 1.5, rtol=1e-10)

    def test_idw_inf_distance(self):
        """Test IDW with infinite distance (effectively zero weight)."""
        values = np.array([[10, 20, 30]])
        distances = np.array([[1.0, np.inf, np.inf]])
        result = inverse_distance_weighted_interpolation(values, distances)

        # Only first value should contribute (others have zero weight)
        expected = 10.0
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)


class TestMeshSpacing:
    """Test mesh spacing calculation."""

    def test_mesh_spacing_level_0(self):
        """Test mesh spacing at refinement level 0 (12 points)."""
        spacing = compute_mesh_spacing_km(0)
        # 12 points on Earth: spacing should be ~6500 km
        assert 5000 < spacing < 8000

    def test_mesh_spacing_level_5(self):
        """Test mesh spacing at refinement level 5 (~10k points)."""
        spacing = compute_mesh_spacing_km(5)
        # Level 5 has ~10,242 points, spacing ~220 km
        assert 150 < spacing < 300

    def test_mesh_spacing_decreases_with_level(self):
        """Test that mesh spacing decreases with higher refinement."""
        spacing_4 = compute_mesh_spacing_km(4)
        spacing_5 = compute_mesh_spacing_km(5)
        spacing_6 = compute_mesh_spacing_km(6)

        assert spacing_4 > spacing_5 > spacing_6

    def test_mesh_spacing_halves_per_level(self):
        """Test that spacing roughly halves per refinement level."""
        spacing_4 = compute_mesh_spacing_km(4)
        spacing_5 = compute_mesh_spacing_km(5)

        # Each level multiplies points by 4, so spacing should halve
        ratio = spacing_4 / spacing_5
        np.testing.assert_allclose(ratio, 2.0, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
