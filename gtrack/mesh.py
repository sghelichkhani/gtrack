"""
Icosahedral mesh generation for initial ocean points.

This module provides functions to create uniform spherical meshes
using icosahedral refinement, matching GPlately's approach.
"""

import math
import numpy as np
from typing import Tuple

import pygplates


def _lat_lon_to_cart(lat_rad: float, lon_rad: float) -> np.ndarray:
    """Convert latitude/longitude (radians) to unit Cartesian coordinates."""
    cos_lat = math.cos(lat_rad)
    return np.array([
        cos_lat * math.cos(lon_rad),
        cos_lat * math.sin(lon_rad),
        math.sin(lat_rad)
    ])


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _distance_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points (lat/lon in radians)."""
    return math.acos(
        math.sin(lat1) * math.sin(lat2) +
        math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    )


def _find_neighbours(point: np.ndarray, all_points: np.ndarray) -> list:
    """Find the 5 adjacent vertices for a given icosahedron vertex."""
    min_distance = None
    neighbours = []
    for idx, p in enumerate(all_points):
        dist = _distance_rad(
            math.radians(point[0]), math.radians(point[1]),
            math.radians(p[0]), math.radians(p[1])
        )
        if dist > 1e-6:  # Skip self (floating-point may give tiny non-zero distance)
            if min_distance is not None and math.isclose(min_distance, dist, rel_tol=1e-9, abs_tol=1e-3):
                neighbours.append(idx)
            elif min_distance is None or dist < min_distance:
                min_distance = dist
                neighbours = [idx]
    assert len(neighbours) in (5, 6)
    return neighbours


def _find_faces(vertices: np.ndarray) -> list:
    """Return 20 icosahedron faces for the given vertices ([lat, lon] in degrees)."""
    faces = []
    for idx, p in enumerate(vertices):
        neighbours = _find_neighbours(p, vertices)
        for i in range(len(neighbours)):
            dist_i_p = _distance_rad(
                math.radians(vertices[neighbours[i]][0]),
                math.radians(vertices[neighbours[i]][1]),
                math.radians(p[0]), math.radians(p[1])
            )
            for j in range(len(neighbours)):
                if i >= j:
                    continue
                dist_i_j = _distance_rad(
                    math.radians(vertices[neighbours[i]][0]),
                    math.radians(vertices[neighbours[i]][1]),
                    math.radians(vertices[neighbours[j]][0]),
                    math.radians(vertices[neighbours[j]][1])
                )
                if math.isclose(dist_i_j, dist_i_p, rel_tol=1e-9, abs_tol=1e-3):
                    faces.append(sorted([idx, neighbours[i], neighbours[j]]))

    unique_faces = []
    for f in faces:
        if f not in unique_faces:
            unique_faces.append(f)
    assert len(unique_faces) == 20
    return unique_faces


def _get_vertices_and_faces_stripy() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return vertices and faces matching Stripy's icosahedral mesh.

    This ensures compatibility with GPlately's mesh generation.
    Reference: https://github.com/underworldcode/stripy
    """
    mid_lat = np.degrees(np.arctan(0.5))  # 26.56505117707799 degrees
    vertices_latlon = np.array([
        [90, 0.0],
        [mid_lat, 0.0],
        [-mid_lat, 36.0],
        [mid_lat, 72.0],
        [-mid_lat, 108.0],
        [mid_lat, 144.0],
        [-mid_lat, 180.0],
        [mid_lat, -72.0],
        [-mid_lat, -36.0],
        [mid_lat, -144.0],
        [-mid_lat, -108.0],
        [-90, 0.0],
    ])

    vertices_xyz = np.array([
        _lat_lon_to_cart(math.radians(v[0]), math.radians(v[1]))
        for v in vertices_latlon
    ])

    return vertices_xyz, np.array(_find_faces(vertices_latlon))


def _deduplicate_vertices(vertices: np.ndarray, tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate vertices using rounding for numerical stability.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex coordinates, shape (N, 3).
    tolerance : float
        Tolerance for considering vertices as duplicates.

    Returns
    -------
    unique_vertices : np.ndarray
        Unique vertices, shape (M, 3) where M <= N.
    inverse_indices : np.ndarray
        Mapping from original to unique indices, shape (N,).
    """
    # Round coordinates for comparison
    rounded = np.round(vertices / tolerance).astype(np.int64)

    # Use structured array for efficient unique finding
    dtype = [('x', np.int64), ('y', np.int64), ('z', np.int64)]
    structured = np.zeros(len(rounded), dtype=dtype)
    structured['x'] = rounded[:, 0]
    structured['y'] = rounded[:, 1]
    structured['z'] = rounded[:, 2]

    _, unique_idx, inverse_idx = np.unique(
        structured, return_index=True, return_inverse=True
    )

    return vertices[unique_idx], inverse_idx


def _bisect_one_level(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform one level of mesh bisection (vectorized).

    Each triangular face is subdivided into 4 smaller triangles by
    adding midpoints on each edge.

    Parameters
    ----------
    vertices : np.ndarray
        Current vertices, shape (V, 3).
    faces : np.ndarray
        Current faces as vertex indices, shape (F, 3).

    Returns
    -------
    new_vertices : np.ndarray
        Vertices after refinement (deduplicated).
    new_faces : np.ndarray
        Faces after refinement.
    """
    n_faces = len(faces)
    n_verts = len(vertices)

    # Get all face vertices at once (vectorized)
    v0 = vertices[faces[:, 0]]  # (n_faces, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute all midpoints at once (vectorized)
    m01 = (v0 + v1) * 0.5  # midpoint of edge 0-1
    m12 = (v1 + v2) * 0.5  # midpoint of edge 1-2
    m20 = (v2 + v0) * 0.5  # midpoint of edge 2-0

    # Normalize all midpoints to unit sphere (vectorized)
    m01 = m01 / np.linalg.norm(m01, axis=1, keepdims=True)
    m12 = m12 / np.linalg.norm(m12, axis=1, keepdims=True)
    m20 = m20 / np.linalg.norm(m20, axis=1, keepdims=True)

    # Stack all vertices: original + new midpoints
    # Midpoints will have duplicates at shared edges
    all_verts = np.vstack([vertices, m01, m12, m20])

    # Deduplicate vertices
    unique_verts, inverse = _deduplicate_vertices(all_verts)

    # Map original vertex indices through inverse mapping
    idx_v0 = inverse[faces[:, 0]]
    idx_v1 = inverse[faces[:, 1]]
    idx_v2 = inverse[faces[:, 2]]

    # Map midpoint indices through inverse mapping
    # m01 starts at index n_verts
    # m12 starts at index n_verts + n_faces
    # m20 starts at index n_verts + 2*n_faces
    idx_m01 = inverse[n_verts + np.arange(n_faces)]
    idx_m12 = inverse[n_verts + n_faces + np.arange(n_faces)]
    idx_m20 = inverse[n_verts + 2 * n_faces + np.arange(n_faces)]

    # Create 4 new faces per old face (vectorized)
    # Face layout after subdivision:
    #       v0
    #      /  \
    #    m20--m01
    #    / \  / \
    #  v2--m12--v1
    new_faces = np.vstack([
        np.column_stack([idx_v0, idx_m01, idx_m20]),   # top triangle
        np.column_stack([idx_m01, idx_v1, idx_m12]),   # right triangle
        np.column_stack([idx_m20, idx_m12, idx_v2]),   # left triangle
        np.column_stack([idx_m01, idx_m12, idx_m20]),  # center triangle
    ])

    return unique_verts, new_faces


def _bisect(vertices: np.ndarray, faces: np.ndarray, level: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recursively bisect icosahedron faces to refine the mesh.

    Uses vectorized operations and vertex deduplication for efficiency.
    Each level multiplies the number of faces by 4.

    Parameters
    ----------
    vertices : np.ndarray
        Initial vertices, shape (V, 3).
    faces : np.ndarray
        Initial faces as vertex indices, shape (F, 3).
    level : int
        Number of refinement levels.

    Returns
    -------
    vertices : np.ndarray
        Refined vertices.
    faces : np.ndarray
        Refined faces.
    """
    for _ in range(level):
        vertices, faces = _bisect_one_level(vertices, faces)
    return vertices, faces


def _xyz_to_lonlat(x: float, y: float, z: float) -> Tuple[float, float]:
    """Convert unit Cartesian coordinates to longitude/latitude in degrees."""
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)
    return np.rad2deg(lon), np.rad2deg(lat)


def create_icosahedral_mesh(refinement_levels: int = 5) -> pygplates.MultiPointOnSphere:
    """
    Create an icosahedral mesh as pygplates.MultiPointOnSphere.

    This matches GPlately's mesh generation for compatibility.
    The mesh starts as a 12-vertex icosahedron and is recursively
    refined by bisecting each face.

    Parameters
    ----------
    refinement_levels : int, default=5
        Number of refinement levels. Higher values create denser meshes.
        Level 5 produces ~10,242 points.
        Level 6 produces ~40,962 points.

    Returns
    -------
    pygplates.MultiPointOnSphere
        The mesh points as a MultiPointOnSphere object.

    Examples
    --------
    >>> mesh = create_icosahedral_mesh(refinement_levels=5)
    >>> len(list(mesh.get_points()))
    10242
    """
    vertices, faces = _get_vertices_and_faces_stripy()
    refined_vertices, _ = _bisect(vertices, faces, refinement_levels)

    # Convert XYZ to lat/lon and create MultiPointOnSphere
    points = []
    for v in refined_vertices:
        lon, lat = _xyz_to_lonlat(v[0], v[1], v[2])
        # pygplates expects (lat, lon) order
        points.append((lat, lon))

    return pygplates.MultiPointOnSphere(points)


def create_icosahedral_mesh_latlon(refinement_levels: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an icosahedral mesh returning latitude and longitude arrays.

    Parameters
    ----------
    refinement_levels : int, default=5
        Number of refinement levels.

    Returns
    -------
    lats : np.ndarray
        Latitudes in degrees, shape (N,)
    lons : np.ndarray
        Longitudes in degrees, shape (N,)

    Examples
    --------
    >>> lats, lons = create_icosahedral_mesh_latlon(refinement_levels=5)
    >>> len(lats)
    10242
    """
    vertices, faces = _get_vertices_and_faces_stripy()
    refined_vertices, _ = _bisect(vertices, faces, refinement_levels)

    lats = np.zeros(len(refined_vertices))
    lons = np.zeros(len(refined_vertices))

    for i, v in enumerate(refined_vertices):
        lon, lat = _xyz_to_lonlat(v[0], v[1], v[2])
        lats[i] = lat
        lons[i] = lon

    return lats, lons


def create_icosahedral_mesh_xyz(refinement_levels: int = 5, radius: float = 1.0) -> np.ndarray:
    """
    Create an icosahedral mesh returning XYZ coordinates.

    Parameters
    ----------
    refinement_levels : int, default=5
        Number of refinement levels.
    radius : float, default=1.0
        Radius of the sphere (1.0 for unit sphere, or Earth radius in meters).

    Returns
    -------
    xyz : np.ndarray
        XYZ coordinates, shape (N, 3)

    Examples
    --------
    >>> xyz = create_icosahedral_mesh_xyz(refinement_levels=5)
    >>> xyz.shape
    (10242, 3)
    """
    vertices, faces = _get_vertices_and_faces_stripy()
    refined_vertices, _ = _bisect(vertices, faces, refinement_levels)

    return refined_vertices * radius


def mesh_point_count(refinement_levels: int) -> int:
    """
    Calculate the number of points in an icosahedral mesh.

    The formula is: 10 * 4^levels + 2

    Parameters
    ----------
    refinement_levels : int
        Number of refinement levels.

    Returns
    -------
    int
        Number of points in the mesh.

    Examples
    --------
    >>> mesh_point_count(5)
    10242
    >>> mesh_point_count(6)
    40962
    """
    return 10 * (4 ** refinement_levels) + 2
