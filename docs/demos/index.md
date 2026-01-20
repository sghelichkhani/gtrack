# Demos

These demos show how to use gtrack for computing lithospheric structure through geological time.

## Seafloor Age Tracking

- **[Simple Example](seafloor_age_simple.ipynb)**: One-shot computation of present-day seafloor ages using the `compute_ages()` class method.

- **[Stepwise Example](seafloor_age_stepwise.ipynb)**: Stepwise evolution through geological time with visualisation at intermediate ages. Demonstrates checkpointing and the incremental API.

## Continental Rotation

- **[Continental Rotation](continental_rotation.ipynb)**: Rotate user-provided continental points through geological time using plate reconstructions.

## Running the Demos

The demos require plate model data. To download the data and run the examples locally:

```bash
cd examples
make data      # Download plate model data
make notebooks # Generate and execute notebooks
```

The plate model used is Matthews et al. (2016), available from the [gadopt data server](https://data.gadopt.org/demos/).
