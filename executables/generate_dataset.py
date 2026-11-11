import numpy as np

from sim import simulate
from pathlib import Path

VAL_SPLIT = 0.01734

ROOM_DIM = [70, 70, 3]
MARGIN = 1.0                # [m] minimum distance of the sources from the walls
Z = 1.0                     # [m] height of sensors and sources, the same for both (the localization is 2D)
VAL_GRID_SIDE = 16         # validation grid of VAL_GRID_SIDE x VAL_GRID_SIDE positions
TEST_GRID_SIDE = 16         # test grid of TEST_GRID_SIDE x TEST_GRID_SIDE positions, even: with an odd side
                            # a node is in the centre of the room, on the central sensor of the Y
FS = 16000
SEED = 0

ROOT = Path(__file__).parent.parent
VALIDATION_DIR = ROOT / "data" / "validation"
TRAINING_DIR = ROOT / "data" / "training"
TEST_DIR = ROOT / "data" / "test"
SIGNAL_POOL = ROOT / "sim" / "test_signals" / "pool"

# Same formations of the old campaign, with the centre of the array in the centre of the room (35, 35).
# The aperture is the maximum distance between two sensors, both formations are rotated by 10 deg:
# Y: one sensor in the centre and three at 120 deg (100, 220, 340 deg) at distance aperture / sqrt(3)
# square: vertices at 235, 325, 55, 145 deg at distance aperture / 2 (half diagonal)
# Name: <layout>_<aperture>m_<SNR>db, with p instead of the decimal point (3p5m = 3.5 m)
#SENSOR_POS = {
#    "y_3p5m_10db": np.array([[35.000, 35.000, Z], [34.649, 36.990, Z], [33.452, 33.701, Z], [36.899, 34.309, Z]]),
#    "y_7m_10db": np.array([[35.000, 35.000, Z], [34.298, 38.980, Z], [31.904, 32.402, Z], [38.798, 33.618, Z]]),
#    "y_10m_10db": np.array([[35.000, 35.000, Z], [33.997, 40.686, Z], [30.577, 31.289, Z], [40.425, 33.025, Z]]),
#    "sq_3p5m_10db": np.array([[33.996, 33.566, Z], [36.434, 33.996, Z], [36.004, 36.434, Z], [33.566, 36.004, Z]]),
#    "sq_7m_10db": np.array([[32.992, 32.133, Z], [37.867, 32.992, Z], [37.008, 37.867, Z], [32.133, 37.008, Z]]),
#    "sq_10m_10db": np.array([[32.132, 30.904, Z], [39.096, 32.132, Z], [37.868, 39.096, Z], [30.904, 37.868, Z]]),
#}

SENSOR_POS = {
    "y_7m_10db": np.array([[35.000, 35.000, Z], [34.298, 38.980, Z], [31.904, 32.402, Z], [38.798, 33.618, Z]]),
}

def grid(n_side: int, centered: bool) -> np.ndarray:
    """
    Grid of n_side x n_side source positions (n_side**2, 3) inside the room, at least MARGIN from the walls.
    centered = False: the nodes include the borders of the area, centered = True: the nodes are the centres of
    n_side x n_side cells. A centered and a not centered grid never have a node in common.
    """
    if(centered):
        step = (ROOM_DIM[0] - 2 * MARGIN) / n_side
        coords = MARGIN + step * (np.arange(n_side) + 0.5)
    else:
        coords = np.linspace(MARGIN, ROOM_DIM[0] - MARGIN, n_side)
    gx, gy = np.meshgrid(coords, coords, indexing="ij")
    return np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, Z)))

def make_dataset(out_dir: Path, sensor_pos: np.ndarray, source_pos: np.ndarray, snr_db: float, seed: int) -> None:
    """
    Simulate one dataset, skipped if it already exists (dataset.json is written only at the end of a simulation).
    """
    if((out_dir / "dataset.json").is_file()):
        print(f"Skip {out_dir.relative_to(ROOT)}: already present")
        return
    print(f"Generating {out_dir.relative_to(ROOT)}")
    # overwrite=True only replaces an interrupted simulation, a complete one is skipped above
    simulate(out_dir, sensor_pos, source_pos, FS, snr_db, room=tuple(ROOM_DIM), seed=seed, 
             overwrite=True, signal_pool=SIGNAL_POOL)

def generate_val_datasets() -> None:
    """
    Fill up the validation dataset folder, using a fixed number of simulations.
    """
    source_pos = grid(VAL_GRID_SIDE, centered=True)
    for conf, sensor_pos in SENSOR_POS.items():
        snr_db = int(conf.rsplit("_", 1)[-1].split("db")[0])
        make_dataset(VALIDATION_DIR / conf, sensor_pos, source_pos, snr_db, SEED + 1)

def generate_train_datasets() -> None:
    """
    Generate train dataset using the existing configurations and generating random points
    for the source position, iterating to have the pre-set validation-training split.
    """
    training_dim = round((1 - VAL_SPLIT) * VAL_GRID_SIDE**2 / VAL_SPLIT)
    rng = np.random.default_rng(SEED)
    source_pos = np.full((training_dim, 3), Z)
    source_pos[:, 0] = rng.uniform(MARGIN, ROOM_DIM[0] - MARGIN, training_dim)
    source_pos[:, 1] = rng.uniform(MARGIN, ROOM_DIM[1] - MARGIN, training_dim)
    for conf, sensor_pos in SENSOR_POS.items():
        snr_db = int(conf.rsplit("_", 1)[-1].split("db")[0])
        make_dataset(TRAINING_DIR / conf, sensor_pos, source_pos, snr_db, SEED + 2)

def generate_test_datasets() -> None:
    """
    Same but for test data
    """
    source_pos = grid(TEST_GRID_SIDE, centered=False)
    for conf, sensor_pos in SENSOR_POS.items():
        snr_db = int(conf.rsplit("_", 1)[-1].split("db")[0])
        make_dataset(TEST_DIR / conf, sensor_pos, source_pos, snr_db, SEED + 3)

def main() -> None:
    generate_val_datasets()
    generate_train_datasets()
    generate_test_datasets()

if __name__ == "__main__":
    main()
