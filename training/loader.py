import numpy as np
import torch
import tomllib
import csv

from pathlib import Path
from torch.utils.data import Dataset
from localization.cnn_attention import SELDNet, encode_pos

ROOT = Path(__file__).parent.parent
TRAIN_CONF_PATH = ROOT / "configs" / "train.toml"

class FootstepDataset(Dataset):
    """
    Dataset used to train the CnnAttention neural network for localization.
    Every element is a window of input_size samples of one recording and its target:
    x: (N, input_size) filtered signal of the sensors,
    y: (T, 3) for every frame (dx, dy, d / dist_scale) if the frame is active, (0, 0, 0) otherwise.
    A frame is active if more than frame_thr of its samples are active.
    T is the number of frames of the output of the network for an input of input_size samples.
    """
    def __init__(self, data_dir: Path, input_size: int, window_stride: int, frame_thr: float, dist_scale: float, T: int):
        super().__init__()
        self.input_size = input_size
        self.frame_thr = frame_thr
        self.dist_scale = dist_scale
        self.T = T
        self.frame_size = input_size // T
        self.signals = []
        self.active = []
        self.source_pos = []
        self.arr_center = []
        self.windows = []       # (recording index, first sample of the window)

        datasets = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and (d / "index.csv").is_file()]
        if(len(datasets) == 0):
            print(f"No datasets found in {data_dir}")
        for d in datasets:
            self.load_dataset(d)
        for r in range(len(self.signals)):
            L = self.signals[r].shape[1]
            for start in range(0, L - input_size + 1, window_stride):
                self.windows.append((r, start))

    def load_dataset(self, dataset_dir: Path) -> None:
        """
        Load all the recordings listed in index.csv.
        """
        with open(dataset_dir / "index.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            data = np.load(dataset_dir / row["file"])
            self.signals.append(data["signal"])                                 # (N, L)
            self.active.append(data["active"])                                  # (L,)
            self.source_pos.append(data["source_pos"][:2])                      # (2,)
            self.arr_center.append(data["sensor_pos"][:, :2].mean(axis=0))      # (2,)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        r, start = self.windows[k]
        x = self.signals[r][:, start:start + self.input_size]                  # (N, input_size)
        a = self.active[r][start:start + self.T * self.frame_size]             # (T * frame_size,)
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
    Check of the validation dataset: shapes of one element and fraction of active frames.
    """
    with open(TRAIN_CONF_PATH, "rb") as f:
        conf = tomllib.load(f)
    win = conf["window"]
    T = output_frames(4, win["input_size"])
    dataset = FootstepDataset(ROOT / conf["data"]["val_dir"], win["input_size"], win["window_stride"],
                              win["frame_thr"], win["dist_scale"], T)
    print(f"T = {T}, windows: {len(dataset)}")
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
