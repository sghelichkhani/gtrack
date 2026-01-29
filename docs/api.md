# API Reference

## Seafloor Age Tracking

::: gtrack.SeafloorAgeTracker
    options:
      members:
        - __init__
        - initialize
        - initialize_from_cloud
        - step_to
        - get_current_state
        - save_checkpoint
        - load_checkpoint
        - compute_ages

::: gtrack.TracerConfig

---

## Point Rotation

::: gtrack.PointCloud
    options:
      members:
        - __init__
        - from_latlon
        - xyz
        - latlon
        - lonlat
        - add_property
        - get_property
        - remove_property
        - has_property
        - property_names
        - subset
        - copy

::: gtrack.PointRotator

::: gtrack.PolygonFilter

---

## I/O Functions

::: gtrack.load_points_numpy

::: gtrack.load_points_latlon

::: gtrack.save_points_numpy

::: gtrack.save_points_latlon

::: gtrack.PointCloudCheckpoint

---

## Mesh Generation

::: gtrack.create_sphere_mesh_xyz

::: gtrack.create_sphere_mesh_latlon

---

## Logging

::: gtrack.enable_verbose

::: gtrack.enable_debug

::: gtrack.disable_logging

::: gtrack.set_log_level
