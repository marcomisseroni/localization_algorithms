import argparse
import numpy as np

from pathlib import Path
from core.data import Sample
from algorithms.beamforming_gcc_ls import BeamformingGccLs 

def parse_arguments() -> Path:
    parser = argparse.ArgumentParser(description="Test the implemented algorithms to localize a source having N sensors")
    parser.add_argument('signal_path')
    args = parser.parse_args()
    return Path(args.signal_path)

def run_sample(sample: Sample) -> np.ndarray:
    L = 64
    Nw = 4096
    hop = 512
    sensor_pos = sample.sensors_pos
    fs = sample.fs
    c = sample.c
    algo = BeamformingGccLs(L, Nw, sensor_pos, fs, c)
    est_pos = []
    block = sample.read(hop)
    while(block is not None):
        curr_est = algo.push(block)
        if(curr_est is not None):
            est_pos.append(curr_est)
        block = sample.read(hop)
    return np.array(est_pos)

def compute_error(est_pos: np.ndarray, sample: Sample) -> np.ndarray:
    true_pos = sample.source_pos[:2]
    return np.linalg.norm(est_pos - true_pos, axis=1)

def print_report(err: np.ndarray) -> None:
    if err.size == 0:
        print("No estimates produced")
        return

    print("=" * 40)
    print(" Localization error report")
    print("=" * 40)
    print(f" {'Estimates':<20}{err.size:>12d}")
    print("-" * 40)
    print(f" {'Mean':<20}{err.mean():>10.3f} m")
    print(f" {'Median':<20}{np.median(err):>10.3f} m")
    print(f" {'RMSE':<20}{np.sqrt(np.mean(err**2)):>10.3f} m")
    print(f" {'Std':<20}{err.std():>10.3f} m")
    print(f" {'Min':<20}{err.min():>10.3f} m")
    print(f" {'Max':<20}{err.max():>10.3f} m")
    print("-" * 40)
    for p in (25, 75, 90, 95):
        print(f" {f'P{p}':<20}{np.percentile(err, p):>10.3f} m")
    print("-" * 40)
    for th in (0.05, 0.10, 0.25, 0.50):
        frac = np.mean(err < th) * 100
        print(f" {f'Error < {th * 100:.0f} cm':<20}{frac:>10.1f} %")
    print("=" * 40)

def main():
    path = parse_arguments()
    sample = Sample(path)
    est_pos = run_sample(sample)
    err = compute_error(est_pos, sample)
    print_report(err)

if __name__ == "__main__":
    main()