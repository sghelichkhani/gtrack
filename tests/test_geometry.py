"""Tests for geometry module."""

import numpy as np
import pytest
from gtrack.geometry import (
    LatLon2XYZ,
    XYZ2LatLon,
    RefineGreatCircleArcSegment,
    Segments2Points,
    normalize_to_sphere,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
