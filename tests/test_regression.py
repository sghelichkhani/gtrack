"""
Regression tests for SeafloorAgeTracker.

These tests ensure the scientific output remains consistent across code changes.
Reference data is stored in tests/data/ as npz files.
"""

import numpy as np
import pytest
from pathlib import Path

from tractec import SeafloorAgeTracker, TracerConfig

# Data paths
DATA_DIR = Path(__file__).parent.parent / "data" / "Plate_model"
REF_DIR = Path(__file__).parent / "data"

ROTATION_FILES = [
    DATA_DIR / "Global_EB_250-0Ma_GK07_Matthews++.rot",
    DATA_DIR / "Global_EB_410-250Ma_GK07_Matthews++.rot"
]
TOPOLOGY_FILES = [
    DATA_DIR / "Mesozoic-Cenozoic_plate_boundaries_Matthews++.gpml",
    DATA_DIR / "Paleozoic_plate_boundaries_Matthews++.gpml",
    DATA_DIR / "TopologyBuildingBlocks_Matthews++.gpml",
]
CONTINENTAL_POLYGONS = DATA_DIR / "ContPolys/PresentDay_ContinentalPolygons_Matthews++.shp"


def _data_files_exist():
    """Check if GPlates data files exist."""
    return all(f.exists() for f in ROTATION_FILES) and all(f.exists() for f in TOPOLOGY_FILES)


def _run_tracker_200_to_180():
    """Run tracker from 200 Ma to 180 Ma with 1 Myr timesteps."""
    config = TracerConfig(
        time_step=1.0,
        default_refinement_levels=5,
        initial_ocean_mean_spreading_rate=75.0,
        ridge_sampling_degrees=2.0,
        spreading_offset_degrees=0.01,
        velocity_delta_threshold=7.0,
        distance_threshold_per_myr=10.0,
    )

    tracker = SeafloorAgeTracker(
        rotation_files=ROTATION_FILES,
        topology_files=TOPOLOGY_FILES,
        continental_polygons=CONTINENTAL_POLYGONS,
        config=config,
        verbose=False
    )

    tracker.initialize(starting_age=200)

    # Only store results at key checkpoints (190 and 180 Ma)
    results = {}
    for target_age in [190, 180]:
        cloud = tracker.step_to(target_age)
        results[target_age] = {
            'xyz': cloud.xyz.copy(),
            'ages': cloud.get_property('age').copy(),
        }

    return results


def generate_reference_data():
    """Generate and save reference data. Run manually when needed."""
    REF_DIR.mkdir(parents=True, exist_ok=True)
    results = _run_tracker_200_to_180()

    for age, data in results.items():
        np.savez(REF_DIR / f"ref_age_{age:03d}.npz", xyz=data['xyz'], ages=data['ages'])

    print(f"Saved reference data for ages 190 and 180 to {REF_DIR}")


@pytest.mark.skipif(not _data_files_exist(), reason="GPlates data files not found")
def test_tracker_200_to_180_regression():
    """Test that tracker output matches reference data at key checkpoints."""
    ref_file = REF_DIR / "ref_age_180.npz"
    if not ref_file.exists():
        pytest.skip("Reference data not found. Run generate_reference_data() first.")

    results = _run_tracker_200_to_180()

    for age in [190, 180]:
        ref = np.load(REF_DIR / f"ref_age_{age:03d}.npz")
        np.testing.assert_allclose(results[age]['xyz'], ref['xyz'], rtol=1e-10,
                                   err_msg=f"XYZ mismatch at {age} Ma")
        np.testing.assert_allclose(results[age]['ages'], ref['ages'], rtol=1e-10,
                                   err_msg=f"Ages mismatch at {age} Ma")


if __name__ == "__main__":
    generate_reference_data()
