import numpy as np
import argparse
import tomllib
import csv
import json

from pathlib import Path
from core.data import Sample
from detection.amp_thr import AmpThr

ROOT = Path(__file__).parent.parent
DATASET_PATH_DEFAULT = ROOT / "data" / "training"
CONFIG_PATH = ROOT / "configs" / "default.toml"
OUT_DIR = ROOT / "training" / "data" / "training"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate .npz files used to train neural networks")
    parser.add_argument("--data", default=DATASET_PATH_DEFAULT, 
                        help="Path of the directory containing the datasets generated using the simulator")
    parser.add_argument("--conf", default=CONFIG_PATH, help="Path of the configuration file defalut.toml")
    parser.add_argument("--out", default=OUT_DIR, help="Path of the directory where the dataset will be stored")
    return parser.parse_args()

class DatasetBuilder():
    """
    Class used to generate a suitable dataset to train the CnnAttention neural network for localization.
    """
    def __init__(self, dataset_path: Path, config_path: Path, out_dir: Path):
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.out_dir = out_dir

    def load_conf(self) -> dict:
        with open(self.config_path, "rb") as f:
            data = tomllib.load(f)
        return data

def main() -> None:
    args = parse_arguments()
    dataset_path = Path(args.data)
    config_path = Path(args.conf)
    out_path = Path(args.out)
    dataset = DatasetBuilder(dataset_path, config_path, out_path)
    conf = dataset.load_conf()
    hop = conf["stream"]["hop"]
    datasets = [d for d in sorted(dataset.dataset_path.iterdir()) if d.is_dir() and (d / "manifest.json").is_file()]
    if(len(datasets) == 0):
        print(f"No datasets found in {dataset.dataset_path}")
        return
    for j, d in enumerate(datasets):
        with open(d / "manifest.json") as f:
            manifest = json.load(f)
        
        out_dir = dataset.out_dir / d.name 
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []

        for i, entry in enumerate(manifest["samples"]):
            print(f"Dataset {j + 1}/{len(datasets)} ({d.name}): recording {i + 1}/{len(manifest['samples'])}")
            sample = Sample(d / entry["metadata_file"], filter_order=conf["band"]["filter_order"], 
                                    low_cutoff=conf["band"]["low_cutoff"], high_cutoff=conf["band"]["high_cutoff"])
            det = AmpThr(fs=sample.fs, N=sample.N, pre_margin=conf["event"]["pre_margin"], **conf["detection"]["amp_thr"])
            signal = []
            mask = []
            block = sample.read(hop)
            while(block is not None):
                mask.append(det.compute_mask(block))
                signal.append(block)
                block = sample.read(hop)
            signal = np.concatenate(signal, axis=1)
            signal = signal.astype(np.float32)
            mask = np.concatenate(mask)
            if(signal.shape[1] != mask.shape[0]):
                print("Error: different shape of signal and mask")
                continue
            file_name = Path(entry["metadata_file"]).stem + ".npz"
            np.savez(out_dir / file_name, signal=signal, active=mask, source_pos=sample.source_pos, sensor_pos=sample.sensors_pos, fs=sample.fs)
            dist = np.linalg.norm(sample.source_pos[:2] - sample.sensors_pos[:, :2].mean(axis=0))
            rows.append([file_name, signal.shape[1], mask.mean(),
                         sample.source_pos[0], sample.source_pos[1], dist])
        with open(out_dir / "index.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "length", "active_ratio", "source_x", "source_y", "dist"])
            writer.writerows(rows)

        # Parameters used to generate the .npz files, the network must be used with the same filter
        params = {
            "band": conf["band"],
            "hop": hop,
            "pre_margin": conf["event"]["pre_margin"],
            "amp_thr": conf["detection"]["amp_thr"],
            "signal_dtype": "float32",
        }
        with open(out_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

if __name__ == "__main__":
    main()