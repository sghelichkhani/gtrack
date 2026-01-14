"""
Example: Integrating TracTec into HPC Forward-Time Geodynamic Simulation

This example demonstrates how to use the HPCSeafloorAgeTracker in a
forward-time geodynamic simulation running on HPC systems.

Key features:
- Stateful tracking (maintains tracers in memory)
- Incremental updates (only evolves when needed)
- Checkpointing for restarts
- Minimal overhead for HPC workflows
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tractec.hpc_integration import HPCSeafloorAgeTracker
from tractec import TracerConfig


################################################################################
#                          Configuration
################################################################################

# Plate reconstruction data
folder_path = Path("./../data/mathews2016/")
rotation_files = [folder_path / "Matthews_etal_GPC_2016_410-0Ma_GK07.rot"]
topology_files = [
    folder_path / "Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies.gpmlz",
    folder_path / "Matthews_etal_GPC_2016_Paleozoic_PlateTopologies.gpmlz",
]
continent_polygons = "/Users/sghelichkhani/Workplace/pygplates-tutorials/data/workshop/ContinentalPolygons/Seton_etal_ESR2012_ContinentalPolygons_2012.1.gpmlz"

# Simulation parameters
INITIAL_TIME = 200.0  # Ma - simulation start
FINAL_TIME = 0.0      # Ma - simulation end (present)
SIM_TIMESTEP = 1.0    # Myr - your geodynamic simulation timestep
UPDATE_FREQUENCY = 5  # Update ages every 5 Myr (or when plates reorganize)

# Output
output_dir = Path("hpc_output")
output_dir.mkdir(exist_ok=True)
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)


################################################################################
#                    Simulated Geodynamic Model
#           (Replace this with your actual HPC simulation)
################################################################################

class MockGeodynamicSimulation:
    """
    Mock geodynamic simulation for demonstration.
    Replace this with your actual HPC simulation code.
    """

    def __init__(self, initial_time):
        self.time = initial_time
        self.seafloor_ages = None
        self.plate_config_changed = False

    def step(self, dt):
        """Advance simulation by dt."""
        self.time -= dt  # Forward in time toward present

        # Simulate plate reorganization events
        # In real simulation, this would be detected from your dynamics
        if int(self.time) % 20 == 0:  # Example: reorganize every 20 Myr
            self.plate_config_changed = True
        else:
            self.plate_config_changed = False

    def needs_age_update(self, update_frequency):
        """Determine if ages need updating."""
        # Update if plate configuration changed or at regular intervals
        return (self.plate_config_changed or
                int(self.time) % update_frequency == 0)

    def use_seafloor_ages(self, lons, lats, ages):
        """
        Use seafloor ages in simulation.
        Replace with your actual usage.
        """
        self.seafloor_ages = {
            'lons': lons,
            'lats': lats,
            'ages': ages
        }

        # Example: Compute some property that depends on age
        # In real simulation, you might:
        # - Update thermal boundary condition
        # - Compute lithospheric thickness
        # - Set material properties
        # etc.

        mean_age = np.mean(ages)
        print(f"  Using {len(ages)} seafloor ages (mean: {mean_age:.1f} Ma)")


################################################################################
#                      Main HPC Simulation Loop
################################################################################

def run_hpc_simulation():
    """
    Main simulation loop showing HPC integration.
    """

    print("=" * 70)
    print("HPC Geodynamic Simulation with Seafloor Age Tracking")
    print("=" * 70)

    # =========================================================================
    # INITIALIZATION (Run once at start of HPC job)
    # =========================================================================

    print("\n[1] Initializing seafloor age tracker...")

    # Create tracker with optimized settings
    age_tracker = HPCSeafloorAgeTracker(
        rotation_files=[str(f) for f in rotation_files],
        topology_files=[str(f) for f in topology_files],
        continental_polygons=str(continent_polygons),
        initial_time=INITIAL_TIME,
        max_time=FINAL_TIME,
        config=TracerConfig(
            ridge_resolution=50e3,
            subduction_resolution=20e3,
            time_step=1.0
        ),
        preload_boundaries=True,  # Pre-load for efficiency
        verbose=True
    )

    # Initialize tracers at starting time
    print(f"\n[2] Initializing tracers at {INITIAL_TIME} Ma...")
    num_tracers = age_tracker.initialize_at_time(INITIAL_TIME)
    print(f"  Ready with {num_tracers} tracers")

    # =========================================================================
    # YOUR GEODYNAMIC SIMULATION
    # =========================================================================

    print(f"\n[3] Starting simulation loop...")
    print(f"  Time range: {INITIAL_TIME} Ma → {FINAL_TIME} Ma")
    print(f"  Timestep: {SIM_TIMESTEP} Myr")
    print(f"  Age update frequency: {UPDATE_FREQUENCY} Myr")

    # Initialize your geodynamic model
    geo_sim = MockGeodynamicSimulation(INITIAL_TIME)

    # Main time loop
    iteration = 0
    checkpoint_interval = 50  # Checkpoint every 50 Myr

    while geo_sim.time > FINAL_TIME:
        iteration += 1

        print(f"\n--- Iteration {iteration}: {geo_sim.time:.1f} Ma ---")

        # =====================================================================
        # YOUR SIMULATION STEP
        # =====================================================================

        # Advance your geodynamic simulation
        geo_sim.step(SIM_TIMESTEP)

        # =====================================================================
        # UPDATE SEAFLOOR AGES (when needed)
        # =====================================================================

        if geo_sim.needs_age_update(UPDATE_FREQUENCY):
            print(f"  Updating seafloor ages to {geo_sim.time:.1f} Ma...")

            # Get updated ages (incremental evolution from last state)
            age_data = age_tracker.update_to_time(geo_sim.time)

            # Use ages in your simulation
            geo_sim.use_seafloor_ages(
                age_data['lons'],
                age_data['lats'],
                age_data['ages']
            )

            # Print statistics
            stats = age_tracker.get_statistics()
            print(f"  Age statistics: mean={stats['mean_age']:.1f} Ma, "
                  f"max={stats['max_age']:.1f} Ma")
        else:
            print(f"  No age update needed")

        # =====================================================================
        # CHECKPOINTING (for HPC restarts)
        # =====================================================================

        if int(geo_sim.time) % checkpoint_interval == 0:
            checkpoint_file = checkpoint_dir / f"checkpoint_{int(geo_sim.time)}Ma.npy"
            age_tracker.save_checkpoint(str(checkpoint_file))
            print(f"  Saved checkpoint")

        # =====================================================================
        # OUTPUT (optional, for analysis)
        # =====================================================================

        if int(geo_sim.time) % 10 == 0:  # Save data every 10 Myr
            # Get current ages
            current_ages = age_tracker.get_current_ages()

            # Save to file
            output_file = output_dir / f"ages_{int(current_ages['time'])}Ma.npz"
            np.savez(
                output_file,
                lons=current_ages['lons'],
                lats=current_ages['lats'],
                ages=current_ages['ages'],
                time=current_ages['time']
            )
            print(f"  Saved ages to {output_file.name}")

    # =========================================================================
    # FINALIZATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)

    final_stats = age_tracker.get_statistics()
    print(f"\nFinal state at {final_stats['time']} Ma:")
    print(f"  Tracers: {final_stats['count']}")
    print(f"  Mean age: {final_stats['mean_age']:.1f} Ma")
    print(f"  Max age: {final_stats['max_age']:.1f} Ma")


################################################################################
#                      Restart from Checkpoint
################################################################################

def restart_from_checkpoint(checkpoint_file, restart_time):
    """
    Example: Restart simulation from checkpoint.

    Parameters
    ----------
    checkpoint_file : str
        Path to checkpoint file
    restart_time : float
        Time to restart from
    """

    print("=" * 70)
    print(f"Restarting simulation from {restart_time} Ma")
    print("=" * 70)

    # Re-initialize tracker
    age_tracker = HPCSeafloorAgeTracker(
        rotation_files=[str(f) for f in rotation_files],
        topology_files=[str(f) for f in topology_files],
        continental_polygons=str(continent_polygons),
        initial_time=INITIAL_TIME,
        max_time=FINAL_TIME,
        preload_boundaries=True
    )

    # Load checkpoint
    age_tracker.load_checkpoint(checkpoint_file)

    # Verify state
    stats = age_tracker.get_statistics()
    print(f"\nLoaded state:")
    print(f"  Time: {stats['time']} Ma")
    print(f"  Tracers: {stats['count']}")

    # Continue simulation from checkpoint
    print(f"\nContinuing simulation...")

    # Re-initialize your geodynamic model at restart time
    geo_sim = MockGeodynamicSimulation(restart_time)

    # Continue main loop
    # ... (similar to run_hpc_simulation)


################################################################################
#                      Performance Comparison
################################################################################

def compare_approaches():
    """
    Compare stateful vs. on-demand approaches.
    """
    from tractec.hpc_integration import MemoryEfficientSeafloorAgeTracker
    import time

    print("=" * 70)
    print("Performance Comparison")
    print("=" * 70)

    times = [200, 195, 190, 185, 180]  # 5 Myr intervals

    # =========================================================================
    # Approach 1: Stateful (recommended for your use case)
    # =========================================================================

    print("\n[1] Stateful Approach (maintaining state):")

    tracker = HPCSeafloorAgeTracker(
        rotation_files=[str(f) for f in rotation_files],
        topology_files=[str(f) for f in topology_files],
        continental_polygons=str(continent_polygons),
        initial_time=200,
        max_time=0,
        verbose=False
    )

    tracker.initialize_at_time(200)

    start = time.time()
    for t in times[1:]:
        age_data = tracker.update_to_time(t)
    elapsed_stateful = time.time() - start

    print(f"  Time: {elapsed_stateful:.2f} seconds")
    print(f"  Per update: {elapsed_stateful/len(times[1:]):.2f} seconds")

    # =========================================================================
    # Approach 2: On-demand (recompute each time)
    # =========================================================================

    print("\n[2] On-demand Approach (recomputing each time):")

    tracker_mem = MemoryEfficientSeafloorAgeTracker(
        rotation_files=[str(f) for f in rotation_files],
        topology_files=[str(f) for f in topology_files],
        continental_polygons=str(continent_polygons)
    )

    start = time.time()
    for t in times:
        lons, lats, ages = tracker_mem.get_ages_at_time(t, start_time=200)
    elapsed_ondemand = time.time() - start

    print(f"  Time: {elapsed_ondemand:.2f} seconds")
    print(f"  Per update: {elapsed_ondemand/len(times):.2f} seconds")

    # =========================================================================
    # Comparison
    # =========================================================================

    print(f"\nSpeedup: {elapsed_ondemand/elapsed_stateful:.1f}x faster with stateful approach")
    print("\nConclusion: For your forward-time HPC simulation,")
    print("use HPCSeafloorAgeTracker (stateful approach)")


################################################################################
#                              Main
################################################################################

if __name__ == "__main__":

    # Check if data exists
    if not folder_path.exists():
        print(f"\nERROR: Data folder not found: {folder_path}")
        print("Please update folder_path in this script.")
        exit(1)

    # Run main simulation
    print("\n" + "=" * 70)
    print("Example 1: Full HPC Simulation")
    print("=" * 70)
    run_hpc_simulation()

    # # Example: Restart from checkpoint
    # print("\n" + "=" * 70)
    # print("Example 2: Restart from Checkpoint")
    # print("=" * 70)
    # checkpoint_file = "checkpoints/checkpoint_150Ma.npy"
    # if Path(checkpoint_file).exists():
    #     restart_from_checkpoint(checkpoint_file, restart_time=150)

    # # Performance comparison
    # print("\n" + "=" * 70)
    # print("Example 3: Performance Comparison")
    # print("=" * 70)
    # compare_approaches()

    print("\n✅ Examples complete!\n")
