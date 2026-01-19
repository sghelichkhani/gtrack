"""
Point rotation API for rotating user-provided points through geological time.

This module provides the core API for:
- Managing collections of points with associated properties (PointCloud)
- Rotating points between geological ages using plate reconstructions (PointRotator)

The API uses Cartesian XYZ coordinates internally for compatibility with gadopt,
while interfacing with pygplates using lat/lon coordinates.

Terminology:
- geological_age (or just 'age'): Time before present in Ma (0 = present, 100 = 100 Myr ago)
- from_age / to_age: Source and target geological ages for rotation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import warnings

import numpy as np


@dataclass
class PointCloud:
    """
    Container for points with associated properties.

    Stores points in Cartesian XYZ format internally (matches gadopt).
    Properties (lithospheric_depth, etc.) are stored separately from positions.

    Parameters
    ----------
    xyz : np.ndarray
        Cartesian coordinates, shape (N, 3), in meters.
        Points should lie on Earth's surface (radius ~6.3781e6 m).
    properties : dict, optional
        Dictionary mapping property names to arrays of shape (N,).
        Properties are preserved during rotation operations.
    plate_ids : np.ndarray, optional
        Plate IDs for each point, shape (N,). Required for rotation.

    Examples
    --------
    >>> xyz = np.random.randn(1000, 3)
    >>> from tractec.geometry import normalize_to_sphere
    >>> xyz = normalize_to_sphere(xyz)  # Project to Earth's surface
    >>> cloud = PointCloud(xyz=xyz)
    >>> cloud.add_property('lithospheric_depth', np.random.rand(1000) * 100e3)
    """

    xyz: np.ndarray
    properties: Dict[str, np.ndarray] = field(default_factory=dict)
    plate_ids: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate inputs after initialization."""
        # Ensure xyz is a numpy array
        self.xyz = np.asarray(self.xyz)

        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(
                f"xyz must have shape (N, 3), got {self.xyz.shape}"
            )

        n_points = len(self.xyz)

        # Validate properties
        for name, prop in self.properties.items():
            prop = np.asarray(prop)
            if len(prop) != n_points:
                raise ValueError(
                    f"Property '{name}' has {len(prop)} values, expected {n_points}"
                )
            self.properties[name] = prop

        # Validate plate_ids if provided
        if self.plate_ids is not None:
            self.plate_ids = np.asarray(self.plate_ids)
            if len(self.plate_ids) != n_points:
                raise ValueError(
                    f"plate_ids has {len(self.plate_ids)} values, expected {n_points}"
                )

    @property
    def n_points(self) -> int:
        """Number of points in the cloud."""
        return len(self.xyz)

    @property
    def latlon(self) -> np.ndarray:
        """
        Get lat/lon coordinates (computed from XYZ).

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) with [lat, lon] in degrees.
            Latitude: -90 to 90, Longitude: -180 to 180.
        """
        from .geometry import XYZ2LatLon
        lats, lons = XYZ2LatLon(self.xyz)
        return np.column_stack([lats, lons])

    @property
    def lonlat(self) -> np.ndarray:
        """
        Get lon/lat coordinates (computed from XYZ).

        Returns
        -------
        np.ndarray
            Array of shape (N, 2) with [lon, lat] in degrees.
            Longitude: -180 to 180, Latitude: -90 to 90.
        """
        from .geometry import XYZ2LatLon
        lats, lons = XYZ2LatLon(self.xyz)
        return np.column_stack([lons, lats])

    @classmethod
    def from_latlon(
        cls,
        latlon: np.ndarray,
        properties: Optional[Dict[str, np.ndarray]] = None
    ) -> "PointCloud":
        """
        Create PointCloud from lat/lon coordinates.

        Parameters
        ----------
        latlon : np.ndarray
            Coordinates, shape (N, 2) with [lat, lon] in degrees.
        properties : dict, optional
            Properties to attach to the points.

        Returns
        -------
        PointCloud
            New PointCloud with XYZ coordinates computed from lat/lon.

        Examples
        --------
        >>> latlon = np.array([[45.0, -120.0], [30.0, 90.0]])
        >>> cloud = PointCloud.from_latlon(latlon)
        """
        from .geometry import LatLon2XYZ
        latlon = np.asarray(latlon)
        xyz = LatLon2XYZ(latlon)
        return cls(xyz=xyz, properties=properties or {})

    def add_property(self, name: str, values: np.ndarray) -> None:
        """
        Add or update a property.

        Parameters
        ----------
        name : str
            Name of the property.
        values : np.ndarray
            Property values, shape (N,).

        Raises
        ------
        ValueError
            If values length doesn't match number of points.
        """
        values = np.asarray(values)
        if len(values) != self.n_points:
            raise ValueError(
                f"Property '{name}' has {len(values)} values, expected {self.n_points}"
            )
        self.properties[name] = values

    def remove_property(self, name: str) -> None:
        """
        Remove a property.

        Parameters
        ----------
        name : str
            Name of the property to remove.
        """
        if name in self.properties:
            del self.properties[name]

    def get_property(self, name: str) -> np.ndarray:
        """
        Get a property by name.

        Parameters
        ----------
        name : str
            Name of the property.

        Returns
        -------
        np.ndarray
            Property values.

        Raises
        ------
        KeyError
            If property not found.
        """
        if name not in self.properties:
            raise KeyError(f"Property '{name}' not found")
        return self.properties[name]

    def subset(self, mask: np.ndarray) -> "PointCloud":
        """
        Create subset of points using boolean mask.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask, shape (N,). True values are kept.

        Returns
        -------
        PointCloud
            New PointCloud with subset of points.
        """
        mask = np.asarray(mask, dtype=bool)
        new_xyz = self.xyz[mask]
        new_properties = {
            name: prop[mask] for name, prop in self.properties.items()
        }
        new_plate_ids = self.plate_ids[mask] if self.plate_ids is not None else None
        return PointCloud(
            xyz=new_xyz,
            properties=new_properties,
            plate_ids=new_plate_ids
        )

    def copy(self) -> "PointCloud":
        """
        Create a deep copy.

        Returns
        -------
        PointCloud
            Deep copy of this PointCloud.
        """
        return PointCloud(
            xyz=self.xyz.copy(),
            properties={k: v.copy() for k, v in self.properties.items()},
            plate_ids=self.plate_ids.copy() if self.plate_ids is not None else None
        )

    def __len__(self) -> int:
        """Return number of points."""
        return self.n_points

    def __repr__(self) -> str:
        """String representation."""
        props = list(self.properties.keys())
        has_plate_ids = self.plate_ids is not None
        return (
            f"PointCloud(n_points={self.n_points}, "
            f"properties={props}, "
            f"has_plate_ids={has_plate_ids})"
        )


class PointRotator:
    """
    Rotate points between geological ages using plate reconstructions.

    This class provides the main API for rotating user-provided points
    according to plate tectonic reconstructions.

    Key Features:
    - Cartesian XYZ internal representation (matches gadopt)
    - Properties stored separately from positions and preserved during rotation
    - Batched rotation for performance (10-50x speedup by grouping by plate_id)
    - Handles undefined plates with warnings

    Parameters
    ----------
    rotation_files : list of str
        Paths to rotation model files (.rot).
    topology_files : list of str, optional
        Paths to topology/plate boundary files (.gpmlz).
        Used for partitioning if static_polygons not provided.
    static_polygons : str, optional
        Path to static polygons for plate ID assignment.
        If None, uses topology_files for partitioning.

    Examples
    --------
    >>> rotator = PointRotator(
    ...     rotation_files=['rotations.rot'],
    ...     topology_files=['topologies.gpmlz'],
    ...     static_polygons='static_polygons.gpmlz'
    ... )
    >>>
    >>> # Load user points
    >>> cloud = PointCloud.from_latlon(my_latlon_array)
    >>>
    >>> # Assign plate IDs at present day
    >>> cloud = rotator.assign_plate_ids(cloud, at_age=0.0)
    >>>
    >>> # Rotate to 50 Ma
    >>> rotated = rotator.rotate(cloud, from_age=0.0, to_age=50.0)
    """

    def __init__(
        self,
        rotation_files: Union[str, List[str]],
        topology_files: Optional[Union[str, List[str]]] = None,
        static_polygons: Optional[str] = None
    ):
        import pygplates

        # Ensure rotation_files is a list
        if isinstance(rotation_files, str):
            rotation_files = [rotation_files]

        self.rotation_model = pygplates.RotationModel(rotation_files)

        # Load topology features if provided
        if topology_files is not None:
            if isinstance(topology_files, str):
                topology_files = [topology_files]

            self.topology_features = pygplates.FeatureCollection()
            for file in topology_files:
                features = pygplates.FeatureCollection(file)
                self.topology_features.add(features)
        else:
            self.topology_features = None

        # Load static polygons if provided
        if static_polygons is not None:
            self.static_polygons = pygplates.FeatureCollection(static_polygons)
        else:
            self.static_polygons = None

        # Ensure we have some features for partitioning
        if self.static_polygons is None and self.topology_features is None:
            raise ValueError(
                "Either topology_files or static_polygons must be provided "
                "for plate ID assignment."
            )

    def assign_plate_ids(
        self,
        cloud: PointCloud,
        at_age: float,
        use_static_polygons: bool = True,
        remove_undefined: bool = True
    ) -> PointCloud:
        """
        Assign plate IDs to points based on their positions.

        Parameters
        ----------
        cloud : PointCloud
            Points to assign plate IDs to.
        at_age : float
            Geological age at which to assign plate IDs (Ma).
            Use 0.0 for present-day positions.
        use_static_polygons : bool, default=True
            If True and static_polygons are loaded, use them for assignment.
            Otherwise use topology_features.
        remove_undefined : bool, default=True
            If True, remove points with undefined plate IDs (plate_id=0)
            and emit a warning.
            If False, keep points with plate_id=0.

        Returns
        -------
        PointCloud
            Cloud with plate_ids assigned. May have fewer points if
            remove_undefined=True and some points had undefined plates.

        Warns
        -----
        UserWarning
            If any points have undefined plate IDs.
        """
        from .plate_operations import get_plate_ids

        # Create tracer-like array for compatibility with existing function
        tracers = np.zeros((cloud.n_points, 4))
        tracers[:, :3] = cloud.xyz

        # Select partitioning features
        partitioning_features = (
            self.static_polygons
            if (use_static_polygons and self.static_polygons is not None)
            else self.topology_features
        )

        plate_ids = get_plate_ids(
            tracers,
            partitioning_features,
            self.rotation_model,
            at_age
        )

        # Check for undefined plates (plate_id = 0)
        undefined_mask = (plate_ids == 0)
        n_undefined = np.sum(undefined_mask)

        if n_undefined > 0:
            action = "These points will be removed." if remove_undefined else "They will be assigned plate_id=0."
            warnings.warn(
                f"{n_undefined} points ({100*n_undefined/cloud.n_points:.1f}%) "
                f"have undefined plate IDs at {at_age} Ma. {action}",
                UserWarning
            )

        if remove_undefined and n_undefined > 0:
            valid_mask = ~undefined_mask
            result = cloud.subset(valid_mask)
            result.plate_ids = plate_ids[valid_mask]
        else:
            result = cloud.copy()
            result.plate_ids = plate_ids

        return result

    def rotate(
        self,
        cloud: PointCloud,
        from_age: float,
        to_age: float,
        reassign_plate_ids: bool = False
    ) -> PointCloud:
        """
        Rotate points from one geological age to another.

        Uses batched rotation operations for performance (10-50x speedup
        by grouping points by plate_id).

        Parameters
        ----------
        cloud : PointCloud
            Points to rotate. Must have plate_ids assigned.
        from_age : float
            Source geological age (Ma).
        to_age : float
            Target geological age (Ma).
        reassign_plate_ids : bool, default=False
            If True, reassign plate IDs at target age.
            If False, keep original plate IDs.

        Returns
        -------
        PointCloud
            Rotated points with same properties.

        Raises
        ------
        ValueError
            If cloud does not have plate_ids assigned.

        Notes
        -----
        Direction of rotation:
        - from_age=0, to_age=50: Rotate present-day positions to 50 Ma
        - from_age=50, to_age=0: Rotate 50 Ma positions to present day

        Examples
        --------
        >>> # Rotate present-day continental points to 50 Ma
        >>> rotated = rotator.rotate(cloud, from_age=0.0, to_age=50.0)
        """
        if cloud.plate_ids is None:
            raise ValueError(
                "Cloud must have plate_ids assigned. "
                "Call assign_plate_ids() first."
            )

        from .plate_operations import move_tracers_batched

        # Create tracer array for compatibility with existing function
        tracers = np.zeros((cloud.n_points, 4))
        tracers[:, :3] = cloud.xyz

        # Compute time step (to_age - from_age)
        dt = to_age - from_age

        # Apply batched rotation (the CRITICAL OPTIMIZATION)
        rotated_tracers = move_tracers_batched(
            tracers,
            self.rotation_model,
            cloud.plate_ids,
            dt,
            from_age
        )

        # Create result cloud with rotated positions
        result = PointCloud(
            xyz=rotated_tracers[:, :3],
            properties={k: v.copy() for k, v in cloud.properties.items()},
            plate_ids=cloud.plate_ids.copy()
        )

        # Optionally reassign plate IDs at new age
        if reassign_plate_ids:
            result = self.assign_plate_ids(result, at_age=to_age)

        return result

    def rotate_incremental(
        self,
        cloud: PointCloud,
        from_age: float,
        to_age: float,
        time_step: float = 1.0,
        reassign_at_each_step: bool = True
    ) -> PointCloud:
        """
        Rotate points incrementally through geological time.

        Useful when plate IDs may change during rotation (e.g.,
        points crossing plate boundaries over long time spans).

        Parameters
        ----------
        cloud : PointCloud
            Points to rotate. Must have plate_ids assigned.
        from_age : float
            Source geological age (Ma).
        to_age : float
            Target geological age (Ma).
        time_step : float, default=1.0
            Time step for incremental rotation (Myr).
        reassign_at_each_step : bool, default=True
            If True, reassign plate IDs after each time step.
            This handles points that cross plate boundaries.

        Returns
        -------
        PointCloud
            Rotated points.

        Examples
        --------
        >>> # Rotate with 1 Myr steps, reassigning plates
        >>> rotated = rotator.rotate_incremental(
        ...     cloud, from_age=0.0, to_age=100.0, time_step=1.0
        ... )
        """
        if cloud.plate_ids is None:
            raise ValueError("Cloud must have plate_ids assigned.")

        result = cloud.copy()
        current_age = from_age
        direction = 1 if to_age > from_age else -1

        while (direction > 0 and current_age < to_age) or \
              (direction < 0 and current_age > to_age):

            # Calculate step size (don't overshoot)
            remaining = abs(to_age - current_age)
            actual_step = min(time_step, remaining) * direction
            next_age = current_age + actual_step

            # Rotate one step
            result = self.rotate(
                result,
                from_age=current_age,
                to_age=next_age,
                reassign_plate_ids=reassign_at_each_step
            )

            current_age = next_age

        return result
