import numpy as np
import hashlib
import shutil
import torch
import tomllib
import json

from pathlib import Path
from torch.utils.data import Dataset
from localization.cnn_attention import SELDNet, encode_pos, normalize_input
from core.data import Sample
from detection.amp_thr import AmpThr

ROOT = Path(__file__).parent.parent
TRAIN_CONF_PATH = ROOT / "configs" / "train.toml"
DEFAULT_CONF_PATH = ROOT / "configs" / "default.toml"
CACHE_DIR = "cache"             # subfolder of the dataset with the filtered signals and the masks
CACHE_INFO = "cache.json"       # parameters the cache was computed with and length of every recording, written last

class FootstepDataset(Dataset):
    """
    Dataset used to train the CnnAttention neural network for localization.
    data_dir is the folder of one dataset (with dataset.json), so of one sensor configuration: one network is trained
    for every configuration, since the network does not receive the sensor positions.
    Every recording is read with Sample, so it is filtered as in evaluate.py, and labeled sample by sample as
    active or not with AmpThr.compute_mask, block by block as in streaming. def_toml is the content of default.toml
    (band, stream hop, amp_thr parameters and pre_margin).
    The filtered signal and the mask of every recording are computed once and saved in data_dir/cache/ (.npy);
    every window is then read from disk with a memory map, so the RAM does not grow with the size of the dataset.
    The cache is computed again if dataset.json or the parameters of def_toml that change signal and mask differ.
    Every element is a window of input_size samples of one recording and its target:
    x: (N, input_size) filtered signal of the sensors,
    y: (T, 3) for every frame (dx, dy, d / dist_scale) if the frame is active, (0, 0, 0) otherwise.
    A frame is active if more than frame_thr of its samples are active.
    T is the number of frames of the output of the network for an input of input_size samples.
    """
    def __init__(self, data_dir: Path, input_size: int, window_stride: int, frame_thr: float, dist_scale: float, T: int, def_toml: dict):
        super().__init__()
        self.input_size = input_size
        self.frame_thr = frame_thr
        self.dist_scale = dist_scale
        self.T = T
        self.frame_size = input_size // T
        self.def_toml = def_toml
        self.cache_dir = data_dir / CACHE_DIR
        self.rec_names = []     # stem of the WAV of every kept recording, name of its cache files
        self.lengths = []       # L of every kept recording
        self.source_pos = []
        self.arr_center = []
        self.windows = []       # (recording index, first sample of the window)

        if(not (data_dir / "dataset.json").is_file()):
            raise FileNotFoundError(f"{data_dir} is not a dataset: dataset.json not found")
        self.load_dataset(data_dir)
        for r in range(len(self.lengths)):
            for start in range(0, self.lengths[r] - input_size + 1, window_stride):
                self.windows.append((r, start))

    def cache_params(self, dataset_dir: Path) -> dict:
        """
        Everything the cached signals and masks depend on: content of dataset.json and parameters of def_toml.
        """
        with open(dataset_dir / "dataset.json", "rb") as f:
            json_hash = hashlib.sha256(f.read()).hexdigest()
        return {
            "dataset_json_sha256": json_hash,
            "band": self.def_toml["band"],
            "hop": self.def_toml["stream"]["hop"],
            "pre_margin": self.def_toml["event"]["pre_margin"],
            "amp_thr": self.def_toml["detection"]["amp_thr"],
        }

    def load_dataset(self, dataset_dir: Path) -> None:
        """
        Read the list of the recordings in dataset.json and their length from the cache, building the cache first
        if it is missing, incomplete or computed with other parameters.
        """
        with open(dataset_dir / "dataset.json", "r") as f:
            data = json.load(f)
        arr_center = np.array(data["sensor_pos"])[:, :2].mean(axis=0)                  # (2,)
        params = self.cache_params(dataset_dir)
        info_path = self.cache_dir / CACHE_INFO
        info = None
        if(info_path.is_file()):
            with open(info_path, "r") as f:
                info = json.load(f)
        if(info is None or info["params"] != params):
            info = self.build_cache(dataset_dir, data, params)
        else:
            print(f"Loading {dataset_dir.name}: cache in {self.cache_dir}")
        lengths = info["lengths"]       # file -> L, only the kept recordings
        for rec in data["recordings"]:
            if(rec["file"] not in lengths):
                continue
            self.rec_names.append(Path(rec["file"]).stem)
            self.lengths.append(lengths[rec["file"]])
            self.source_pos.append(np.array(rec["source_pos"][:2]))                    # (2,)
            self.arr_center.append(arr_center)                                         # (2,)

    def build_cache(self, dataset_dir: Path, data: dict, params: dict) -> dict:
        """
        Filtered signal (N, L) float32 and activity mask (L,) of every recording in the cache folder, then
        CACHE_INFO with the parameters and the lengths, written last so that an interrupted cache is built again.
        Returns the content of CACHE_INFO.
        """
        if(self.cache_dir.exists()):
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir()
        hop = params["hop"]
        band = params["band"]
        lengths = {}
        for i, rec in enumerate(data["recordings"]):
            print(f"Caching {dataset_dir.name}: recording {i + 1}/{len(data['recordings'])}", end="\r")
            sample = Sample(dataset_dir / "dataset.json", i, band["low_cutoff"], band["high_cutoff"], band["filter_order"])
            det = AmpThr(fs=sample.fs, N=sample.N, pre_margin=params["pre_margin"], **params["amp_thr"])
            signal = []
            mask = []
            block = sample.read(hop)
            while(block is not None):
                mask.append(det.compute_mask(block))
                signal.append(block)
                block = sample.read(hop)
            if(len(signal) == 0):
                print(f"\nSkip {rec['file']}: shorter than one block of {hop} samples")
                continue
            name = Path(rec["file"]).stem
            signal = np.concatenate(signal, axis=1).astype(np.float32)                 # (N, L), L multiple of hop
            np.save(self.cache_dir / f"{name}_signal.npy", signal)
            np.save(self.cache_dir / f"{name}_mask.npy", np.concatenate(mask))         # (L,)
            lengths[rec["file"]] = signal.shape[1]
        print()
        info = {"params": params, "lengths": lengths}
        with open(self.cache_dir / CACHE_INFO, "w") as f:
            json.dump(info, f, indent=4)
        return info

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, start = self.windows[k]
        # Memory maps: only the samples of the window are read from disk
        signal = np.load(self.cache_dir / f"{self.rec_names[r]}_signal.npy", mmap_mode="r")    # (N, L)
        mask = np.load(self.cache_dir / f"{self.rec_names[r]}_mask.npy", mmap_mode="r")        # (L,)
        x = normalize_input(signal[:, start:start + self.input_size])          # (N, input_size) same normalization of CNNAttention
        a = mask[start:start + self.T * self.frame_size]                       # (T * frame_size,)
        frame_active = a.reshape(self.T, self.frame_size).mean(axis=1) > self.frame_thr     # (T,)
        y = np.zeros((self.T, 3))
        for t in range(self.T):
            y[t, :] = encode_pos(self.source_pos[r], frame_active[t], self.arr_center[r], self.dist_scale)
        return torch.from_numpy(x), torch.from_numpy(y).float()                # (N, input_size), (T, 3)

def output_frames(N: int, input_size: int) -> int:
    """
    Number of frames T of the output of the network for an input of input_size samples.
    """
    with torch.no_grad():
        y = SELDNet(N).eval()(torch.zeros(1, N, input_size))                   # (1, T, 3)
    return y.shape[1]

def main() -> None:
    """
    Check of the first validation dataset: shapes of one element and fraction of active frames.
    """
    with open(TRAIN_CONF_PATH, "rb") as f:
        conf = tomllib.load(f)
    with open(DEFAULT_CONF_PATH, "rb") as f:
        def_toml = tomllib.load(f)
    win = conf["window"]
    val_dir = ROOT / conf["data"]["val_dir"]
    datasets = [d for d in sorted(val_dir.iterdir()) if (d / "dataset.json").is_file()]
    if(len(datasets) == 0):
        print(f"No datasets found in {val_dir}")
        return
    with open(datasets[0] / "dataset.json") as f:
        N = len(json.load(f)["sensor_pos"])
    T = output_frames(N, win["input_size"])
    dataset = FootstepDataset(datasets[0], win["input_size"], win["window_stride"],
                              win["frame_thr"], win["dist_scale"], T, def_toml)
    print(f"Dataset {datasets[0].name}: T = {T}, windows: {len(dataset)}")
    x, y = dataset[0]
    print(f"x: {tuple(x.shape)} {x.dtype}, y: {tuple(y.shape)} {y.dtype}")
    print(y)

    n_active = 0
    for k in range(len(dataset)):
        y = dataset[k][1]
        n_active += int((y[:, 2] > 0).sum())
    print(f"Active frames: {100 * n_active / (len(dataset) * T):.1f} %")

if __name__ == "__main__":
    main()
