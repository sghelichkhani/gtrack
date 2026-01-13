"""
Polygon filtering operations for point clouds.

This module provides functionality to filter points based on polygon containment,
useful for selecting points inside continental polygons, exclusion zones, etc.
"""

from typing import List, Union
import warnings

import numpy as np


class PolygonFilter:
    """
    Filter points by polygon containment.

    Supports filtering by:
    - Continental polygons (keep only continental points)
    - Custom polygons (user-provided)
    - Exclusion zones (remove points inside polygons)

    Parameters
    ----------
    polygon_files : str or list of str
        Path(s) to polygon files (.gpml, .gpmlz).
    rotation_files : str or list of str
        Paths to rotation model files (.rot).

    Examples
    --------
    >>> filter = PolygonFilter(
    ...     polygon_files='continental_polygons.gpmlz',
    ...     rotation_files=['rotations.rot']
    ... )
    >>>
    >>> # Keep only continental points
    >>> continental_cloud = filter.filter_inside(cloud, at_age=0.0)
    >>>
    >>> # Remove continental points (keep oceanic)
    >>> oceanic_cloud = filter.filter_outside(cloud, at_age=0.0)
    """

    def __init__(
        self,
        polygon_files: Union[str, List[str]],
        rotation_files: Union[str, List[str]]
    ):
        import pygplates

        # Handle single file as list
        if isinstance(polygon_files, str):
            polygon_files = [polygon_files]
        if isinstance(rotation_files, str):
            rotation_files = [rotation_files]

        self.rotation_model = pygplates.RotationModel(rotation_files)

        # Load polygon features
        self.polygon_features = pygplates.FeatureCollection()
        for file in polygon_files:
            features = pygplates.FeatureCollection(file)
            self.polygon_features.add(features)

    def get_containment_mask(
        self,
        cloud: "PointCloud",
        at_age: float
    ) -> np.ndarray:
        """
        Get boolean mask of points inside polygons.

        Parameters
        ----------
        cloud : PointCloud
            Points to check.
        at_age : float
            Geological age at which to check containment (Ma).
            Use 0.0 for present-day polygons.

        Returns
        -------
        np.ndarray
            Boolean mask, shape (N,), True for points inside polygons.
        """
        import pygplates
        from .geometry import XYZ2LatLon

        # Convert to lat/lon for pygplates
        lats, lons = XYZ2LatLon(cloud.xyz)

        # Create partitioner at specified age
        partitioner = pygplates.PlatePartitioner(
            self.polygon_features,
            self.rotation_model,
            reconstruction_time=at_age,
            sort_partitioning_plates=(
                pygplates.SortPartitioningPlates
                .by_partition_type_then_plate_area
            )
        )

        # Check each point
        # Note: pygplates doesn't have batch partition_point, so we loop
        # This is still efficient because PlatePartitioner does internal spatial indexing
        mask = np.zeros(cloud.n_points, dtype=bool)

        for i in range(cloud.n_points):
            point = pygplates.PointOnSphere(lats[i], lons[i])
            # partition_point returns None if not inside any polygon
            mask[i] = partitioner.partition_point(point) is not None

        return mask

    def filter_inside(
        self,
        cloud: "PointCloud",
        at_age: float
    ) -> "PointCloud":
        """
        Keep only points inside polygons.

        Parameters
        ----------
        cloud : PointCloud
            Points to filter.
        at_age : float
            Geological age at which to check containment (Ma).

        Returns
        -------
        PointCloud
            Points inside polygons.

        Examples
        --------
        >>> # Keep only continental points at present day
        >>> continental = filter.filter_inside(cloud, at_age=0.0)
        """
        mask = self.get_containment_mask(cloud, at_age)
        n_inside = np.sum(mask)
        n_total = cloud.n_points

        if n_inside == 0:
            warnings.warn(
                f"No points found inside polygons at {at_age} Ma. "
                f"Returning empty PointCloud.",
                UserWarning
            )

        return cloud.subset(mask)

    def filter_outside(
        self,
        cloud: "PointCloud",
        at_age: float
    ) -> "PointCloud":
        """
        Keep only points outside polygons.

        Parameters
        ----------
        cloud : PointCloud
            Points to filter.
        at_age : float
            Geological age at which to check containment (Ma).

        Returns
        -------
        PointCloud
            Points outside polygons.

        Examples
        --------
        >>> # Remove continental points (keep oceanic) at present day
        >>> oceanic = filter.filter_outside(cloud, at_age=0.0)
        """
        mask = self.get_containment_mask(cloud, at_age)
        n_outside = np.sum(~mask)

        if n_outside == 0:
            warnings.warn(
                f"All points are inside polygons at {at_age} Ma. "
                f"Returning empty PointCloud.",
                UserWarning
            )

        return cloud.subset(~mask)

    def get_statistics(
        self,
        cloud: "PointCloud",
        at_age: float
    ) -> dict:
        """
        Get statistics about polygon containment.

        Parameters
        ----------
        cloud : PointCloud
            Points to analyze.
        at_age : float
            Geological age at which to check containment (Ma).

        Returns
        -------
        dict
            Statistics including:
            - total: Total number of points
            - inside: Number of points inside polygons
            - outside: Number of points outside polygons
            - inside_fraction: Fraction of points inside
        """
        mask = self.get_containment_mask(cloud, at_age)
        n_inside = int(np.sum(mask))
        n_total = cloud.n_points

        return {
            'total': n_total,
            'inside': n_inside,
            'outside': n_total - n_inside,
            'inside_fraction': n_inside / n_total if n_total > 0 else 0.0,
            'at_age': at_age
        }
