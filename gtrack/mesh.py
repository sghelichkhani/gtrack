"""
Sphere mesh generation using Fibonacci spiral distribution.

This module provides functions to create approximately uniform point
distributions on a sphere using the Fibonacci spiral algorithm.
"""

import numpy as np
from typing import Tuple


def create_sphere_mesh_xyz(n_points: int, radius: float = 1.0) -> np.ndarray:
    """
    Create approximately uniform points on a sphere using Fibonacci spiral.

    The Fibonacci spiral algorithm distributes points approximately evenly
    over the surface of a sphere using the golden angle. This avoids the
    pole clustering problem of regular lat/lon grids.

    Parameters
    ----------
    n_points : int
        Number of points to generate on the sphere's surface.
    radius : float, default=1.0
        Radius of the sphere (1.0 for unit sphere, or Earth radius in meters).

    Returns
    -------
    xyz : np.ndarray
        XYZ coordinates, shape (n_points, 3).

    Examples
    --------
    >>> xyz = create_sphere_mesh_xyz(10000)
    >>> xyz.shape
    (10000, 3)
    >>> xyz = create_sphere_mesh_xyz(40000, radius=6.3781e6)  # Earth radius
    >>> xyz.shape
    (40000, 3)
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")

    # Golden angle in radians
    phi = np.pi * (3.0 - np.sqrt(5.0))

    # Generate indices
    indices = np.arange(n_points)

    # Y coordinates: evenly spaced from 1 to -1
    y = 1.0 - (indices / (n_points - 1)) * 2.0 if n_points > 1 else np.array([0.0])

    # Radius at each y (distance from y-axis)
    r = np.sqrt(1.0 - y * y)

    # Angle for each point (golden angle spiral)
    theta = phi * indices

    # Cartesian coordinates on unit sphere
    x = np.cos(theta) * r
    z = np.sin(theta) * r

    # Stack and scale by radius
    xyz = np.column_stack([x, y, z]) * radius

    return xyz


def create_sphere_mesh_latlon(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create approximately uniform points on a sphere returning lat/lon coordinates.

    Parameters
    ----------
    n_points : int
        Number of points to generate.

    Returns
    -------
    lats : np.ndarray
        Latitudes in degrees, shape (n_points,). Range: -90 to 90.
    lons : np.ndarray
        Longitudes in degrees, shape (n_points,). Range: -180 to 180.

    Examples
    --------
    >>> lats, lons = create_sphere_mesh_latlon(10000)
    >>> len(lats)
    10000
    >>> lats.min() >= -90 and lats.max() <= 90
    True
    """
    xyz = create_sphere_mesh_xyz(n_points, radius=1.0)

    # Convert XYZ to lat/lon
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    lats = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    lons = np.degrees(np.arctan2(z, x))

    return lats, lons
