from tractec import HPCSeafloorAgeTracker, TracerConfig

# Initialize the tracker
age_tracker = HPCSeafloorAgeTracker(
    rotation_files=['../data/mathews2016/Matthews_etal_GPC_2016_410-0Ma_GK07.rot'],
    topology_files=['../data/mathews2016/Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies.gpmlz',
                    '../data/mathews2016/Matthews_etal_GPC_2016_Paleozoic_PlateTopologies.gpmlz'],
    continental_polygons="/Users/sghelichkhani/Workplace/pygplates-tutorials/data/workshop/ContinentalPolygons/Seton_etal_ESR2012_ContinentalPolygons_2012.1.gpmlz",
    initial_time=200,  # Your simulation start time
    max_time=0,        # Your simulation end time
    config=TracerConfig(
        ridge_resolution=50e3,
        subduction_resolution=20e3,
        time_step=1.0
    ),
    preload_boundaries=True,  # Cache all boundaries in memory
    verbose=True
)

# Initialize tracers at starting time
age_tracker.initialize_at_time(200)

for time in range(200, 0, -1):
    age_tracker.update_to_time(time)
    # age_tracker.get_current_ages()["xyz"] and "time"