"""
Generate the TOML files of the datasets used to train and validate the neural network localizer.

The sensors are read from simulation/arrays/<ARRAY>.toml (geometry only), the other parameters are the constants
below. Before writing, the array is compared with the test configs of simulation/configs/general/ that use it, so the
network is trained on the same array used in the tests. For every SNR it writes:
    training/<array>_random_snr<SNR>.toml      TRAINING_POSITIONS random source positions inside AREA
    validation/<array>_grid_snr<SNR>.toml      grid VALIDATION_GRID, shifted w.r.t. the 5 m test grid (2.5, 7.5, ...)
                                               so that no validation position is a test position
Every file has its own seed, so positions and noise are different in every dataset.

Usage, from the project root:
    venv/bin/python simulation/generate_nn_configs.py [--array y_7m_center]
then generate every dataset with simulation/generate_dataset.py.
"""

import numpy as np
import argparse
import sys
import tomllib
import tomli_w

from pathlib import Path

ROOT = Path(__file__).parent.parent
ARRAY = "y_7m_center"                       # default array, name of a file of simulation/arrays/
ARRAYS_DIR = ROOT / "simulation" / "arrays"
GENERAL_CONF_DIR = ROOT / "simulation" / "configs" / "general"
TRAINING_CONF_DIR = ROOT / "simulation" / "configs" / "training"
VALIDATION_CONF_DIR = ROOT / "simulation" / "configs" / "validation"

# Same values of the test configs
FS = 16000                                  # [Hz]
C = 343.42                                  # propagation speed [m/s]
MODEL = "free_field"
ROOM = [50.0, 50.0, 3.0]                    # [m]
SIGNAL = "test_signals/footstep_walk.wav"
Z = 1.0                                     # same plane as the sensors (the localization is 2D)

SNRS = [10, 20, 30]                         # [dB] one training and one validation dataset for each
TRAINING_POSITIONS = 1000                   # random source positions of every training dataset
AREA = [[1.0, 49.0], [1.0, 49.0]]           # [[xmin, xmax], [ymin, ymax]] of the random positions [m]
VALIDATION_GRID = [1.25, 46.25, 5.0]        # [start, stop, step] on x and y [m]: 100 nodes
MIN_SENSOR_DIST = 0.6                       # [m] minimum distance between a validation node and a sensor, as in the test grid
SEED_START = 1000

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the training and validation configs of the neural network for one array")
    parser.add_argument("--array", default=ARRAY, help=f"name of a file of simulation/arrays/ (default: {ARRAY})")
    return parser.parse_args()

def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data

def check_array(array: str, sensors: dict) -> None:
    """
    Stop if a test config of the same array (<array>_snr<SNR>.toml) has different sensors.
    """
    for path in sorted(GENERAL_CONF_DIR.glob(f"{array}_snr*.toml")):
        if(load_toml(path)["sensors"] != sensors):
            sys.exit(f"Error: the sensors of {path.relative_to(ROOT)} are different from simulation/arrays/{array}.toml")

def check_grid(array: str, sensors: dict) -> None:
    """
    Stop if a node of the validation grid is closer than MIN_SENSOR_DIST to a sensor.
    The nodes are computed as in simulation/generate_dataset.py (stop included).
    """
    start, stop, step = VALIDATION_GRID
    g = np.arange(start, stop + step / 2, step)
    gx, gy = np.meshgrid(g, g)
    nodes = np.column_stack((gx.ravel(), gy.ravel()))                                   # (n_nodes, 2)
    pos = np.array(sensors["positions"])[:, :2]                                         # (N, 2)
    min_dist = np.min(np.linalg.norm(nodes[:, None, :] - pos[None, :, :], axis=2))
    if(min_dist < MIN_SENSOR_DIST):
        sys.exit(f"Error: a node of the validation grid is {min_dist:.2f} m from a sensor of {array}, "
                 f"less than {MIN_SENSOR_DIST} m: change VALIDATION_GRID")

def make_config(sensors: dict, source: dict, snr: float) -> dict:
    """
    Config of one dataset with the given source positions and SNR.
    """
    return {
        "simulation": {"fs": FS, "c": C, "model": MODEL, "room": ROOM},
        "sensors": sensors,
        "source": {"signal": SIGNAL, "z": Z, **source},
        "noise": {"snr_db": snr},
    }

def write_toml(path: Path, conf: dict) -> None:
    with open(path, "wb") as f:
        tomli_w.dump(conf, f)
    print(path.relative_to(ROOT))

def main() -> None:
    args = parse_arguments()
    array = args.array
    if(not (ARRAYS_DIR / f"{array}.toml").is_file()):
        sys.exit(f"Error: simulation/arrays/{array}.toml does not exist")
    sensors = load_toml(ARRAYS_DIR / f"{array}.toml")["sensors"]
    check_array(array, sensors)
    check_grid(array, sensors)
    TRAINING_CONF_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_CONF_DIR.mkdir(parents=True, exist_ok=True)

    seed = SEED_START
    for snr in SNRS:
        source = {"n_positions": TRAINING_POSITIONS, "seed": seed, "area": AREA}
        write_toml(TRAINING_CONF_DIR / f"{array}_random_snr{snr}.toml", make_config(sensors, source, snr))
        seed += 1

        # With a grid the seed only sets the noise
        source = {"grid_x": VALIDATION_GRID, "grid_y": VALIDATION_GRID, "seed": seed}
        write_toml(VALIDATION_CONF_DIR / f"{array}_grid_snr{snr}.toml", make_config(sensors, source, snr))
        seed += 1

if __name__ == "__main__":
    main()
