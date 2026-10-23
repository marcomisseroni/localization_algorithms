import argparse
import csv
import numpy as np
import tomllib
import json

from pathlib import Path
from core.data import Sample
from localization.base import Localizer
from localization.beamforming_gcc_ls import BeamformingGccLs 
from detection.base import Detector
from detection.amp_thr import AmpThr
from analysis.plot_results import plot_all

LOCALIZERS = {
    "bf_gcc_ls": BeamformingGccLs,
}

DETECTORS= {
    "amp_thr": AmpThr,
}

def parse_arguments() -> Path:
    parser = argparse.ArgumentParser(description="Test the implemented algorithms to localize a source having N sensors")
    parser.add_argument('dataset_path')
    parser.add_argument("--loc", choices=LOCALIZERS.keys(), default="bf_gcc_ls")
    parser.add_argument("--det", choices=DETECTORS.keys(), default="amp_thr")
    parser.add_argument("--out", default="results/batch0.csv")
    return parser.parse_args()

def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data

def run_sample(sample: Sample, loc: Localizer, det: Detector, hop: int) -> np.ndarray:
    sensor_pos = sample.sensors_pos
    fs = sample.fs
    c = sample.c
    est_pos = []
    block = sample.read(hop)
    while(block is not None):
        block_det = det.detect(block)
        if(block_det is not None):
            curr_est = loc.load_block_from_det(block_det)
            if(curr_est is not None):
                est_pos.append(curr_est)
        block = sample.read(hop)
    return np.array(est_pos)

def compute_error(est_pos: np.ndarray, sample: Sample) -> np.ndarray | None:
    if(est_pos.shape[0] == 0):
        return None
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
    args = parse_arguments()
    dataset_path = Path(args.dataset_path)

    LocClass = LOCALIZERS[args.loc]
    DetClass = DETECTORS[args.det]

    conf_path = Path(__file__).parent 
    conf = load_config(conf_path / "configs/default.toml")
    hop = conf["stream"]["hop"]

    with open(dataset_path / "manifest.json") as f:
        manifest = json.load(f)

    n_samples = len(manifest["samples"])
    rows = []           # one row per estimate, or one empty row if the sample has none
    all_err = []
    n_missed = 0

    for i, entry in enumerate(manifest["samples"]):
        sample = Sample(dataset_path / entry["metadata_file"])
        name = Path(entry["metadata_file"]).stem
        true_x, true_y = sample.source_pos[:2]
        # Distance between the source and the centre of the sensor array (mean of the sensor positions)
        dist = np.linalg.norm(sample.source_pos[:2] - sample.sensors_pos[:, :2].mean(axis=0))

        # New objects for every sample, so that no state is carried over
        loc = LocClass(
            sensor_pos=sample.sensors_pos,
            fs=sample.fs,
            c=sample.c,
            **conf["localization"][args.loc]
        )
        det = DetClass(
            fs=sample.fs,
            N=sample.N,
            **conf["detection"][args.det]
        )

        est_pos = run_sample(sample, loc, det, hop)
        err = compute_error(est_pos, sample)
        if(err is None):
            n_missed += 1
            rows.append([name, true_x, true_y, "", "", ""])
            print(f"[{i + 1}/{n_samples}] {name} (distance {dist:.2f} m): no estimate")
            continue
        for p, e in zip(est_pos, err):
            rows.append([name, true_x, true_y, p[0], p[1], e])
        all_err.extend(err)
        print(f"[{i + 1}/{n_samples}] {name} (distance {dist:.2f} m): {len(err)} estimates, median error {np.median(err):.3f} m")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "true_x", "true_y", "est_x", "est_y", "error_m"])
        writer.writerows(rows)
    print(f"Results written to {out_path}")

    print(f"Samples with at least one estimate: {n_samples - n_missed}/{n_samples}")
    print_report(np.array(all_err))

    plot_all(out_path, dataset_path, out_path.parent)

if __name__ == "__main__":
    main()