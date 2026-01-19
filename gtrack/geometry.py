"""
Geometric utility functions for coordinate transformations and operations on a sphere.

All functions are vectorized using numpy for performance.
"""

import numpy as np


# Earth's radius in meters
EARTH_RADIUS = 6.3781e6


def LatLon2XYZ(latlon):
    """
    Convert geographic coordinates (lat, lon) to Cartesian coordinates (x, y, z).

    Parameters
    ----------
    latlon : np.ndarray
        Array of shape (N, 2) containing [latitude, longitude] pairs in degrees.

    Returns
    -------
    xyz : np.ndarray
        Array of shape (N, 3) containing [x, y, z] Cartesian coordinates in meters.

    Examples
    --------
    >>> latlons = np.array([[0, 0], [90, 0], [-90, 0]])
    >>> xyz = LatLon2XYZ(latlons)
    >>> xyz.shape
    (3, 3)
    """
    R = EARTH_RADIUS
    lat = latlon[:, 0]
    lon = latlon[:, 1]
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    x = np.zeros([len(lon), 3])  # initialize coordinate-array
    x[:, 0] = R * np.cos(lat) * np.cos(lon)  # load x-coordinates
    x[:, 1] = R * np.cos(lat) * np.sin(lon)  # load y-coordinates
    x[:, 2] = R * np.sin(lat)  # load z-coordinates
    return x


def XYZ2LatLon(xyz):
    """
    Convert Cartesian coordinates (x, y, z) to geographic coordinates (lat, lon).

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (N, 3) containing [x, y, z] Cartesian coordinates in meters.

    Returns
    -------
    lats : np.ndarray
        Array of latitudes in degrees, shape (N,).
    lons : np.ndarray
        Array of longitudes in degrees, shape (N,).

    Examples
    --------
    >>> xyz = np.array([[6.3781e6, 0, 0], [0, 0, 6.3781e6]])
    >>> lats, lons = XYZ2LatLon(xyz)
    >>> np.allclose(lats, [0, 90])
    True
    >>> np.allclose(lons, [0, 0])
    True
    """
    R = EARTH_RADIUS
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    lats = np.arcsin(z / R) * 360 / (2 * np.pi)
    lons = np.arctan2(y, x) * 360 / (2 * np.pi)
    return lats, lons


def RefineGreatCircleArcSegment(p1, p2, N):
    """
    Refine a great circle arc segment defined by two points into N segments.

    This function takes two points on a sphere and creates a refined path
    between them by interpolating along the great circle arc.

    Parameters
    ----------
    p1 : tuple or list
        First point as (latitude, longitude) in degrees.
    p2 : tuple or list
        Second point as (latitude, longitude) in degrees.
    N : int
        Number of segments to create (N+1 points total).

    Returns
    -------
    lats : np.ndarray
        Array of latitudes along the refined segment, shape (N+1,).
    lons : np.ndarray
        Array of longitudes along the refined segment, shape (N+1,).

    Notes
    -----
    Two points that define a line going through Earth's center will give
    erroneous results.

    Examples
    --------
    >>> p1 = (0, 0)  # Equator at prime meridian
    >>> p2 = (0, 90)  # Equator at 90E
    >>> lats, lons = RefineGreatCircleArcSegment(p1, p2, 2)
    >>> len(lats)
    3
    """
    R = EARTH_RADIUS

    LatLon = np.array([p1, p2])
    XYZ = LatLon2XYZ(LatLon)
    p1 = XYZ[0, :]
    p2 = XYZ[1, :]

    n = np.arange(N + 1)  # weights
    p = np.zeros([N + 1, 3])  #
    for i in range(0, N + 1):
        p[i, :] = (float(N) - n[i]) / N * p1 + (n[i] / float(N)) * p2
    x_mag = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2 + p[:, 2] ** 2)  # compute vector magnitude
    x_mag = np.transpose(np.array([x_mag, x_mag, x_mag]))
    p = (p / x_mag) * R  # project new position back to Earth surface
    lats, lons = XYZ2LatLon(p)
    return lats, lons


def Segments2Points(segments, res):
    """
    Convert great circle arc segments to a set of points with specified resolution.

    Takes line segments defined by endpoint pairs and refines them into a set of
    points spaced at approximately the specified resolution.

    Parameters
    ----------
    segments : np.ndarray
        Array of shape (N, 4) where each row is [lat1, lon1, lat2, lon2] defining
        a segment's endpoints in degrees.
    res : float
        Target resolution for point spacing in meters.

    Returns
    -------
    lats : np.ndarray
        Array of latitudes for all refined points in degrees.
    lons : np.ndarray
        Array of longitudes for all refined points in degrees.

    Examples
    --------
    >>> segments = np.array([[0, 0, 0, 10], [0, 10, 0, 20]])
    >>> lats, lons = Segments2Points(segments, 100e3)  # 100 km resolution
    >>> len(lats) > 2  # Should have more points than segments
    True
    """
    num_segments = len(segments[:, 0])
    lats, lons = np.array([]), np.array([])

    R = EARTH_RADIUS
    # Fixed: Process all segments starting from index 0 (was bug in original starting at 1)
    for i in range(0, num_segments):
        p1, p2 = segments[i, :2], segments[i, 2:]
        lat1, lon1, lat2, lon2 = np.deg2rad(p1[0]), np.deg2rad(p1[1]), np.deg2rad(p2[0]), np.deg2rad(p2[1])
        dlon = abs(lon2 - lon1)
        dist = R * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon))

        if dist > 0:
            N = int(round(dist / res))
        else:
            N = 0

        if N > 0:
            lats_, lons_ = RefineGreatCircleArcSegment(p1, p2, N)
            lats = np.append(lats, lats_)
            lons = np.append(lons, lons_)

    return lats, lons


def normalize_to_sphere(xyz, radius=None):
    """
    Normalize XYZ coordinates to lie on a sphere of given radius.

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (N, 3) containing Cartesian coordinates.
    radius : float, optional
        Radius of the sphere in meters. If None, uses Earth's radius.

    Returns
    -------
    xyz_normalized : np.ndarray
        Array of shape (N, 3) with points projected onto the sphere.

    Examples
    --------
    >>> xyz = np.array([[1, 0, 0], [0, 1, 0]])
    >>> normalized = normalize_to_sphere(xyz, radius=1.0)
    >>> np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    True
    """
    if radius is None:
        radius = EARTH_RADIUS

    magnitudes = np.linalg.norm(xyz, axis=1, keepdims=True)
    return (xyz / magnitudes) * radius
