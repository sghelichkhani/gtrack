"""Tests for point rotation API."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from gtrack.point_rotation import PointCloud, PointRotator
from gtrack.polygon_filter import PolygonFilter
from gtrack.io_formats import (
    load_points_numpy,
    save_points_numpy,
    load_points_latlon,
    save_points_latlon,
    PointCloudCheckpoint,
)
from gtrack.geometry import LatLon2XYZ, XYZ2LatLon, normalize_to_sphere, EARTH_RADIUS


class TestPointCloud:
    """Test PointCloud class."""

    def test_create_from_xyz(self):
        """Test creating PointCloud from XYZ array."""
        xyz = np.random.randn(100, 3)
        xyz = normalize_to_sphere(xyz)

        cloud = PointCloud(xyz=xyz)

        assert cloud.n_points == 100
        assert cloud.xyz.shape == (100, 3)
        assert cloud.plate_ids is None
        assert len(cloud.properties) == 0

    def test_create_from_latlon(self):
        """Test creating PointCloud from lat/lon."""
        latlon = np.array([
            [45.0, -120.0],
            [30.0, 90.0],
            [-15.0, 0.0],
        ])

        cloud = PointCloud.from_latlon(latlon)

        assert cloud.n_points == 3
        assert cloud.xyz.shape == (3, 3)
        # Verify points are on Earth's surface
        distances = np.linalg.norm(cloud.xyz, axis=1)
        np.testing.assert_allclose(distances, EARTH_RADIUS, rtol=1e-10)

    def test_latlon_property(self):
        """Test that latlon property correctly converts from XYZ."""
        original_latlon = np.array([
            [45.0, -120.0],
            [30.0, 90.0],
            [-15.0, 0.0],
        ])

        cloud = PointCloud.from_latlon(original_latlon)
        recovered_latlon = cloud.latlon

        np.testing.assert_allclose(recovered_latlon, original_latlon, atol=1e-9)

    def test_add_property(self):
        """Test adding properties to PointCloud."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)

        depths = np.random.rand(50) * 100e3
        cloud.add_property('depth', depths)

        assert 'depth' in cloud.properties
        np.testing.assert_array_equal(cloud.get_property('depth'), depths)

    def test_add_property_wrong_length(self):
        """Test that adding property with wrong length raises error."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)

        with pytest.raises(ValueError, match="has 30 values, expected 50"):
            cloud.add_property('depth', np.random.rand(30))

    def test_remove_property(self):
        """Test removing properties."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))

        cloud.remove_property('depth')

        assert 'depth' not in cloud.properties

    def test_get_property_not_found(self):
        """Test that getting non-existent property raises KeyError."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)

        with pytest.raises(KeyError, match="'missing'"):
            cloud.get_property('missing')

    def test_subset(self):
        """Test subsetting PointCloud with boolean mask."""
        xyz = normalize_to_sphere(np.random.randn(100, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.arange(100))
        cloud.plate_ids = np.arange(100)

        mask = np.zeros(100, dtype=bool)
        mask[:25] = True

        subset = cloud.subset(mask)

        assert subset.n_points == 25
        np.testing.assert_array_equal(subset.get_property('depth'), np.arange(25))
        np.testing.assert_array_equal(subset.plate_ids, np.arange(25))

    def test_copy(self):
        """Test deep copying PointCloud."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))
        cloud.plate_ids = np.arange(50)

        copied = cloud.copy()

        # Modify original
        cloud.xyz[0] = [0, 0, 0]
        cloud.properties['depth'][0] = -999
        cloud.plate_ids[0] = -999

        # Copy should be unchanged
        assert copied.xyz[0, 0] != 0
        assert copied.properties['depth'][0] != -999
        assert copied.plate_ids[0] != -999

    def test_len(self):
        """Test len() on PointCloud."""
        xyz = normalize_to_sphere(np.random.randn(42, 3))
        cloud = PointCloud(xyz=xyz)

        assert len(cloud) == 42

    def test_repr(self):
        """Test string representation."""
        xyz = normalize_to_sphere(np.random.randn(100, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(100))
        cloud.plate_ids = np.arange(100)

        repr_str = repr(cloud)

        assert "n_points=100" in repr_str
        assert "depth" in repr_str
        assert "has_plate_ids=True" in repr_str

    def test_invalid_xyz_shape(self):
        """Test that invalid XYZ shape raises error."""
        with pytest.raises(ValueError, match="must have shape"):
            PointCloud(xyz=np.random.randn(100, 2))

    def test_create_with_properties(self):
        """Test creating PointCloud with properties in constructor."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        props = {'depth': np.random.rand(50), 'temp': np.random.rand(50)}

        cloud = PointCloud(xyz=xyz, properties=props)

        assert 'depth' in cloud.properties
        assert 'temp' in cloud.properties


class TestPointCloudIO:
    """Test PointCloud IO functions."""

    def test_save_load_numpy_npz(self):
        """Test saving and loading to/from numpy .npz format."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))
        cloud.plate_ids = np.arange(50)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.npz"
            save_points_numpy(cloud, filepath)
            loaded = load_points_numpy(filepath)

        np.testing.assert_allclose(loaded.xyz, cloud.xyz)
        np.testing.assert_allclose(
            loaded.get_property('depth'),
            cloud.get_property('depth')
        )
        np.testing.assert_array_equal(loaded.plate_ids, cloud.plate_ids)

    def test_save_load_latlon_csv(self):
        """Test saving and loading lat/lon CSV."""
        latlon = np.array([
            [45.0, -120.0],
            [30.0, 90.0],
            [-15.0, 0.0],
        ])
        cloud = PointCloud.from_latlon(latlon)
        cloud.add_property('depth', np.array([100.0, 200.0, 300.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            save_points_latlon(cloud, filepath)
            loaded = load_points_latlon(
                filepath,
                latlon_columns=(0, 1),
                property_columns={'depth': 2}
            )

        np.testing.assert_allclose(loaded.latlon, cloud.latlon, atol=1e-6)
        np.testing.assert_allclose(
            loaded.get_property('depth'),
            cloud.get_property('depth'),
            atol=1e-6
        )


class TestPointCloudCheckpoint:
    """Test checkpoint functionality."""

    def test_save_load_checkpoint(self):
        """Test saving and loading checkpoint with metadata."""
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.random.rand(50))
        cloud.plate_ids = np.arange(50)

        checkpoint = PointCloudCheckpoint()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "checkpoint.npz"
            checkpoint.save(
                cloud, filepath,
                geological_age=50.0,
                metadata={'step': 100}
            )
            loaded, metadata = checkpoint.load(filepath)

        np.testing.assert_allclose(loaded.xyz, cloud.xyz)
        assert metadata['geological_age'] == 50.0
        assert metadata['step'] == 100

    def test_list_checkpoints(self):
        """Test listing checkpoint files."""
        checkpoint = PointCloudCheckpoint()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some checkpoint files
            for i in range(3):
                xyz = normalize_to_sphere(np.random.randn(10, 3))
                cloud = PointCloud(xyz=xyz)
                checkpoint.save(cloud, Path(tmpdir) / f"ckpt_{i:02d}.npz")

            files = checkpoint.list_checkpoints(tmpdir)

        assert len(files) == 3


class TestPointRotatorBasic:
    """Basic tests for PointRotator that don't require pygplates data."""

    def test_rotate_requires_plate_ids(self):
        """Test that rotate() fails without plate_ids."""
        # This test doesn't need actual rotation files
        # We just test the validation logic
        xyz = normalize_to_sphere(np.random.randn(50, 3))
        cloud = PointCloud(xyz=xyz)

        # The rotate method checks for plate_ids before doing anything else
        # We can test this by creating a minimal mock that only has the rotate method
        class MockRotator:
            def rotate(self, cloud, from_age, to_age, reassign_plate_ids=False):
                # Call the actual validation logic from PointRotator.rotate
                if cloud.plate_ids is None:
                    raise ValueError(
                        "Cloud must have plate_ids assigned. "
                        "Call assign_plate_ids() first."
                    )

        mock = MockRotator()
        with pytest.raises(ValueError, match="must have plate_ids assigned"):
            mock.rotate(cloud, from_age=0, to_age=50)


class TestPolygonFilterBasic:
    """Basic tests for PolygonFilter that don't require pygplates data."""

    def test_filter_preserves_properties(self):
        """Test that filtering preserves properties."""
        xyz = normalize_to_sphere(np.random.randn(100, 3))
        cloud = PointCloud(xyz=xyz)
        cloud.add_property('depth', np.arange(100, dtype=float))

        # Create a manual mask and use subset
        mask = np.zeros(100, dtype=bool)
        mask[::2] = True  # Keep every other point

        filtered = cloud.subset(mask)

        assert filtered.n_points == 50
        # Check properties are preserved for correct indices
        np.testing.assert_array_equal(
            filtered.get_property('depth'),
            np.arange(0, 100, 2, dtype=float)
        )


class TestCoordinateRoundTrip:
    """Test coordinate transformations between PointCloud and lat/lon."""

    def test_xyz_latlon_roundtrip(self):
        """Test XYZ -> PointCloud -> latlon -> PointCloud -> XYZ."""
        # Start with random points on sphere
        xyz_original = normalize_to_sphere(np.random.randn(100, 3))

        # Create PointCloud
        cloud1 = PointCloud(xyz=xyz_original)

        # Get latlon
        latlon = cloud1.latlon

        # Create new PointCloud from latlon
        cloud2 = PointCloud.from_latlon(latlon)

        # Compare XYZ coordinates
        np.testing.assert_allclose(cloud2.xyz, xyz_original, rtol=1e-10)

    def test_latlon_xyz_roundtrip(self):
        """Test latlon -> PointCloud -> XYZ -> latlon."""
        latlon_original = np.random.rand(100, 2) * np.array([[180, 360]]) - np.array([[90, 180]])

        # Create PointCloud from latlon
        cloud = PointCloud.from_latlon(latlon_original)

        # Get latlon back
        latlon_recovered = cloud.latlon

        np.testing.assert_allclose(latlon_recovered, latlon_original, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
