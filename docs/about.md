# About gtrack

**gtrack** (GPlates-based Tracking) is a Python package for computing lithospheric structure through geological time using plate tectonic reconstructions.

## Key Features

- **Seafloor Age Tracking**: Compute oceanic lithosphere ages using Lagrangian particle tracking
- **Continental Point Rotation**: Rotate user-provided points through geological time
- **GPlately Compatible**: Results match GPlately's SeafloorGrid output
- **gadopt Integration**: Cartesian XYZ coordinates directly compatible with gadopt's interpolation

## Use Cases

gtrack is designed for geodynamic simulations where you need lithospheric structure at past geological ages:

1. **Oceanic lithosphere**: Ages computed via seafloor tracking translate to lithospheric depth via thermal models (e.g., half-space cooling)

2. **Continental lithosphere**: Present-day structure from seismic tomography can be rotated back in time to reconstruct past configurations

## Dependencies

gtrack uses [pygplates](https://www.gplates.org/docs/pygplates/) as the underlying engine for plate reconstructions, with [stripy](https://github.com/underworldcode/stripy) for icosahedral mesh generation.

## License

gtrack is released under the MIT License.

## Author

- S. Ghelichkhani
