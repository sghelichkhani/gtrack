from tractec import SeafloorAgeModel, TracerConfig
from pathlib import Path

# Plate reconstructions to use
folder_path = Path("./../data/mathews2016/")

# Set path to rotation files
rotation_files = [folder_path / "Matthews_etal_GPC_2016_410-0Ma_GK07.rot"]

# Set path to plate boundary files
plate_boundary_files = [
    folder_path / "Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies.gpmlz",
    folder_path / "Matthews_etal_GPC_2016_Paleozoic_PlateTopologies.gpmlz",
    # folder_path / "Matthews_etal_GPC_2016_TopologyBuildingBlocks.gpmlz"
]

# Set path to continental polygon files
continent_polygons = Path((
    "/Users/sghelichkhani/Workplace/pygplates-tutorials/data/workshop/"
    "ContinentalPolygons/Seton_etal_ESR2012_ContinentalPolygons_2012.1.gpmlz"
    )
)

# Create configuration
config = TracerConfig(
    ridge_resolution=50e3,  # 50 km ridge resolution
    subduction_resolution=20e3,  # 20 km subduction resolution
    ridge_offset=1e3,  # 1 km ridge offset
    time_step=1.0  # 1 Myr time step
)

model = SeafloorAgeModel(
    rotation_files=rotation_files,
    topology_files=plate_boundary_files,
    continental_polygons=continent_polygons,
    config=config
)

age_end = 8  # [Myr] Must be an integer smaller than age_beginning
age_beginning = 10  # [Myr] Must be an integer larger than age_end

# Pre-load boundaries for better performance
model.preload_boundaries(range(age_end, age_beginning + 1))
print(f"Boundaries, are pre-loaded for {age_beginning}-{age_end} Ma.")

# Initialise tracers at start time, this needs to be done
# to start the tracers from somewhere
tracers = model._initialize_tracers(age_beginning)
age = age_beginning

while age > age_end:

    # Evolve tracers by one timestep
    tracers = model._evolve_tracers(tracers, age, age - 1)

    # update time
    age -= 1

    print(f"Tracer count: {len(tracers)}")
    print(f"Processed: {age} Ma.")

    # # Save results at specified frequency
    # if next_time % save_freq == 0:
    #     print(f"\nSaving results for {next_time} Ma...")
    #     save_tracers(tracers, next_time, output_dir, save_format)

    #     # Generate age grid and plot
    #     ages, lons, lats = model._tracers_to_grid(tracers, next_time, resolution=1.0)
    #     plot_path = plot_dir / f"age_grid_{next_time}Ma.png"
    #     # plot_age_grid(ages, lons, lats, next_time, plot_path)
