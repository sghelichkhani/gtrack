"""
Spatial indexing for efficient point-in-polygon queries.

This module provides a quad tree spatial index for efficiently testing
many points against many polygons on a sphere. The algorithm groups
nearby points and uses bounding polygon tests to quickly determine
containment for large groups of points at once.

The implementation is based on the algorithm from GPlates Plate Tectonic Tools (ptt),
originally developed at the University of Sydney.
"""

import math
from typing import List, Optional, Sequence, Tuple, Any

import pygplates


# Default subdivision depth for the quad tree.
# A value of 4 works well for point spacings of ~1 degree or less.
DEFAULT_SUBDIVISION_DEPTH = 4


class PointsSpatialTree:
    """
    Quad tree spatial index for points on a sphere.

    Groups points into a hierarchical tree structure based on their
    lat/lon coordinates. This enables efficient spatial queries by
    allowing entire groups of points to be included or excluded
    based on their bounding regions.

    Parameters
    ----------
    points : sequence of pygplates.PointOnSphere
        The points to index.
    subdivision_depth : int, default=4
        Maximum depth of the quad tree. Leaf nodes have lat/lon width
        of 90 / 2^depth degrees. Higher values create finer subdivisions
        but take longer to build.
    """

    def __init__(
        self,
        points: Sequence[pygplates.PointOnSphere],
        subdivision_depth: int = DEFAULT_SUBDIVISION_DEPTH,
    ):
        if subdivision_depth < 0 or subdivision_depth > 100:
            raise ValueError("subdivision_depth must be in range [0, 100]")

        # 8 root nodes covering the globe (2 lat bands x 4 lon quadrants)
        self._root_nodes: List[Optional[_SpatialTreeNode]] = [None] * 8

        # Insert each point into the tree
        for point_index, point in enumerate(points):
            lat, lon = point.to_lat_lon()

            # Determine which root node this point belongs to
            lat_idx = 1 if lat >= 0 else 0
            if lon < -90:
                lon_idx = 0
            elif lon < 0:
                lon_idx = 1
            elif lon < 90:
                lon_idx = 2
            else:
                lon_idx = 3

            root_idx = 4 * lat_idx + lon_idx
            is_north = lat_idx == 1

            # Create root node if needed
            if self._root_nodes[root_idx] is None:
                centre_lon = -180 + 90 * lon_idx + 45
                centre_lat = -90 + 90 * lat_idx + 45
                self._root_nodes[root_idx] = _SpatialTreeNode(
                    centre_lon, centre_lat, 45.0, is_north
                )

            # Traverse down to the appropriate leaf node
            node = self._root_nodes[root_idx]
            node_centre_lon = -180 + 90 * lon_idx + 45
            node_centre_lat = -90 + 90 * lat_idx + 45
            node_half_width = 45.0

            for _ in range(subdivision_depth):
                child_half_width = node_half_width / 2

                if lat < node_centre_lat:
                    child_lat_idx = 0
                    child_centre_lat = node_centre_lat - child_half_width
                else:
                    child_lat_idx = 1
                    child_centre_lat = node_centre_lat + child_half_width

                if lon < node_centre_lon:
                    child_lon_idx = 0
                    child_centre_lon = node_centre_lon - child_half_width
                else:
                    child_lon_idx = 1
                    child_centre_lon = node_centre_lon + child_half_width

                # Create child nodes list if needed
                if node._children is None:
                    node._children = [None] * 4

                child_idx = 2 * child_lat_idx + child_lon_idx
                if node._children[child_idx] is None:
                    node._children[child_idx] = _SpatialTreeNode(
                        child_centre_lon, child_centre_lat, child_half_width, is_north
                    )

                node = node._children[child_idx]
                node_centre_lon = child_centre_lon
                node_centre_lat = child_centre_lat
                node_half_width = child_half_width

            # Add point index to leaf node
            if node._point_indices is None:
                node._point_indices = []
            node._point_indices.append(point_index)

    def get_root_nodes(self) -> List["_SpatialTreeNode"]:
        """Get all non-empty root nodes."""
        return [n for n in self._root_nodes if n is not None]


class _SpatialTreeNode:
    """
    A node in the spatial tree representing a lat/lon region.

    Internal nodes have children; leaf nodes have point indices.
    """

    def __init__(
        self,
        centre_lon: float,
        centre_lat: float,
        half_width: float,
        is_north: bool,
    ):
        self._centre_lon = centre_lon
        self._centre_lat = centre_lat
        self._half_width = half_width
        self._is_north = is_north

        self._children: Optional[List[Optional["_SpatialTreeNode"]]] = None
        self._point_indices: Optional[List[int]] = None
        self._bounding_polygon: Optional[pygplates.PolygonOnSphere] = None

    def is_internal(self) -> bool:
        """True if this is an internal node with children."""
        return self._children is not None

    def is_leaf(self) -> bool:
        """True if this is a leaf node with point indices."""
        return self._point_indices is not None

    def get_children(self) -> List["_SpatialTreeNode"]:
        """Get non-empty child nodes."""
        if self._children is None:
            return []
        return [c for c in self._children if c is not None]

    def get_point_indices(self) -> List[int]:
        """Get point indices in this leaf node."""
        return self._point_indices if self._point_indices else []

    def get_bounding_polygon(self) -> pygplates.PolygonOnSphere:
        """Get polygon bounding this node's region."""
        if self._bounding_polygon is None:
            self._bounding_polygon = self._create_bounding_polygon()
        return self._bounding_polygon

    def _create_bounding_polygon(self) -> pygplates.PolygonOnSphere:
        """Create a polygon that bounds this node's lat/lon region."""
        left_lon = self._centre_lon - self._half_width
        right_lon = self._centre_lon + self._half_width
        bottom_lat = self._centre_lat - self._half_width
        top_lat = self._centre_lat + self._half_width

        points = []

        if self._is_north:
            # Northern hemisphere - need to handle bottom edge carefully
            left_line = pygplates.PolylineOnSphere([(0, left_lon), (90, left_lon)])
            right_line = pygplates.PolylineOnSphere([(0, right_lon), (90, right_lon)])

            bottom_mid = pygplates.PointOnSphere(bottom_lat, (left_lon + right_lon) / 2)

            # Great circle through bottom midpoint oriented toward pole
            axis = pygplates.Vector3D.cross(
                bottom_mid.to_xyz(),
                pygplates.Vector3D.cross(
                    pygplates.PointOnSphere.north_pole.to_xyz(),
                    bottom_mid.to_xyz(),
                ),
            ).to_normalised()
            rotation = pygplates.FiniteRotation(axis.to_xyz(), math.pi / 2)

            bottom_line = pygplates.PolylineOnSphere([
                rotation * bottom_mid,
                bottom_mid,
                rotation.get_inverse() * bottom_mid,
            ])

            _, _, bl = pygplates.GeometryOnSphere.distance(
                bottom_line, left_line, return_closest_positions=True
            )
            _, _, br = pygplates.GeometryOnSphere.distance(
                bottom_line, right_line, return_closest_positions=True
            )

            points = [bl, br,
                      pygplates.PointOnSphere(top_lat, right_lon),
                      pygplates.PointOnSphere(top_lat, left_lon)]
        else:
            # Southern hemisphere - need to handle top edge carefully
            left_line = pygplates.PolylineOnSphere([(0, left_lon), (-90, left_lon)])
            right_line = pygplates.PolylineOnSphere([(0, right_lon), (-90, right_lon)])

            top_mid = pygplates.PointOnSphere(top_lat, (left_lon + right_lon) / 2)

            axis = pygplates.Vector3D.cross(
                top_mid.to_xyz(),
                pygplates.Vector3D.cross(
                    pygplates.PointOnSphere.north_pole.to_xyz(),
                    top_mid.to_xyz(),
                ),
            ).to_normalised()
            rotation = pygplates.FiniteRotation(axis.to_xyz(), math.pi / 2)

            top_line = pygplates.PolylineOnSphere([
                rotation * top_mid,
                top_mid,
                rotation.get_inverse() * top_mid,
            ])

            _, _, tl = pygplates.GeometryOnSphere.distance(
                top_line, left_line, return_closest_positions=True
            )
            _, _, tr = pygplates.GeometryOnSphere.distance(
                top_line, right_line, return_closest_positions=True
            )

            points = [tl, tr,
                      pygplates.PointOnSphere(bottom_lat, right_lon),
                      pygplates.PointOnSphere(bottom_lat, left_lon)]

        return pygplates.PolygonOnSphere(points)


def find_polygons(
    points: Sequence[pygplates.PointOnSphere],
    polygons: Sequence[pygplates.PolygonOnSphere],
    polygon_proxies: Optional[Sequence[Any]] = None,
    all_polygons: bool = False,
    subdivision_depth: int = DEFAULT_SUBDIVISION_DEPTH,
) -> List:
    """
    Find which polygon(s) contain each point.

    Uses a quad tree spatial index for efficient batch queries.
    For uniformly distributed points, this is ~70x faster than
    naive point-by-point testing.

    Parameters
    ----------
    points : sequence of pygplates.PointOnSphere
        Points to test.
    polygons : sequence of pygplates.PolygonOnSphere
        Polygons to test against.
    polygon_proxies : sequence, optional
        Objects to return instead of polygons (e.g., features).
        If None, returns the polygons themselves.
    all_polygons : bool, default=False
        If True, find all polygons containing each point (for overlapping polygons).
        If False, find only the first polygon (for non-overlapping polygons).
    subdivision_depth : int, default=4
        Quad tree subdivision depth.

    Returns
    -------
    list
        For each point: the containing polygon proxy (or None), or
        if all_polygons=True, a list of containing polygon proxies (or None).
    """
    if polygon_proxies is None:
        polygon_proxies = polygons

    if len(polygons) != len(polygon_proxies):
        raise ValueError("polygons and polygon_proxies must have same length")

    # Build spatial tree
    tree = PointsSpatialTree(points, subdivision_depth)

    # Sort polygons by area (largest first) for efficiency
    sorted_pairs = sorted(
        zip(polygons, polygon_proxies),
        key=lambda p: p[0].get_area(),
        reverse=True,
    )

    # Initialize results
    results: List = [None] * len(points)

    # Process each root node
    for root in tree.get_root_nodes():
        _process_node(root, points, sorted_pairs, results, all_polygons)

    return results


def _process_node(
    node: _SpatialTreeNode,
    points: Sequence[pygplates.PointOnSphere],
    polygons_and_proxies: List[Tuple[pygplates.PolygonOnSphere, Any]],
    results: List,
    all_polygons: bool,
) -> None:
    """Recursively process a spatial tree node."""
    # Find polygons that overlap this node
    overlapping = []
    node_poly = node.get_bounding_polygon()

    for polygon, proxy in polygons_and_proxies:
        # Check if node and polygon overlap (distance < threshold means overlap)
        dist = pygplates.GeometryOnSphere.distance(
            node_poly, polygon, 1e-4,
            geometry1_is_solid=True, geometry2_is_solid=True
        )

        if dist is not None:
            # Check if node is completely inside polygon
            outline_dist = pygplates.GeometryOnSphere.distance(
                node_poly, polygon, 1e-4
            )

            completely_inside = False
            if outline_dist is None:
                # Outlines don't touch - check if node is inside polygon
                if polygon.is_point_in_polygon(node_poly.get_exterior_ring_points()[0]):
                    completely_inside = True
                    # Check for interior holes that might be inside node
                    for i in range(polygon.get_number_of_interior_rings()):
                        hole_pt = polygon.get_interior_ring_points(i)[0]
                        if node_poly.is_point_in_polygon(hole_pt):
                            completely_inside = False
                            break

            if completely_inside:
                # Fill entire subtree as inside this polygon
                _fill_node(node, proxy, results, all_polygons)
                if not all_polygons:
                    return
            else:
                overlapping.append((polygon, proxy))

    if not overlapping:
        return

    # Process children or test individual points
    if node.is_internal():
        for child in node.get_children():
            _process_node(child, points, overlapping, results, all_polygons)
    else:
        for idx in node.get_point_indices():
            point = points[idx]
            for polygon, proxy in overlapping:
                if polygon.is_point_in_polygon(point):
                    if all_polygons:
                        if results[idx] is None:
                            results[idx] = []
                        results[idx].append(proxy)
                    else:
                        results[idx] = proxy
                        break


def _fill_node(
    node: _SpatialTreeNode,
    proxy: Any,
    results: List,
    all_polygons: bool,
) -> None:
    """Mark all points in a node's subtree as inside a polygon."""
    if node.is_internal():
        for child in node.get_children():
            _fill_node(child, proxy, results, all_polygons)
    else:
        for idx in node.get_point_indices():
            if all_polygons:
                if results[idx] is None:
                    results[idx] = []
                results[idx].append(proxy)
            else:
                results[idx] = proxy
