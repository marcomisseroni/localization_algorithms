import numpy as np
import json

from scipy.io import wavfile
from pathlib import Path
from scipy import signal

class Sample:
    """
    One recording of a dataset: Sample(path of dataset.json, index of the recording in "recordings", band).
    Reads the common information of the dataset (fs, sensor positions, c, room) and the source position and the
    multichannel WAV of the recording (channel i = sensor i). The signal is read block by block with read(hop),
    band-pass filtered keeping the filter state between blocks, as in streaming.
    data: content of dataset.json already read, to avoid reading the whole file again for every recording of a
    large dataset; if None it is read from dataset_json_path.
    """
    def __init__(self, dataset_json_path: Path, rec_idx: int, low_cutoff: float, high_cutoff: float, filter_order: int,
                 data: dict | None = None) -> None:
        self.data_path = dataset_json_path.parent
        self.cursor = 0
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.filter_order = filter_order

        if(data is None):
            with open(dataset_json_path, "r") as file:
                data = json.load(file)

        self.fs = data["fs"]
        self.rec_idx = rec_idx
        self.rec = data["recordings"][self.rec_idx]
        self.sensors_pos = np.array(data["sensor_pos"])
        self.N = self.sensors_pos.shape[0]
        self.c = data["simulation"]["c"]
        self.room_dim = data["simulation"]["room"]
        self.filt_params = signal.butter(self.filter_order, [self.low_cutoff, self.high_cutoff], btype="bandpass", fs=self.fs, output="sos")
        self.zi = np.zeros((self.filt_params.shape[0], self.N, 2))
        self.source_pos = np.array(self.rec["source_pos"])
        
        path = self.data_path / self.rec["file"]
        _, x = wavfile.read(str(path))
        self.signals = x.T.astype(float)
        if(self.signals.shape[0] != self.N):
            raise ValueError(f"{path} has {self.signals.shape[0]} channels, but {dataset_json_path} has {self.N} sensors")

    def rewind(self) -> None:
        self.cursor = 0
        self.zi = np.zeros((self.filt_params.shape[0], self.N, 2))

    def read(self, hop: int) -> np.ndarray | None:
        """
        Load some data.
        """
        if(self.cursor + hop > self.signals.shape[1]):
            return None
        block = self.signals[:, self.cursor:self.cursor + hop].copy()
        self.cursor += hop
        return self.band_pass_filter(block)

    def band_pass_filter(self, block: np.ndarray) -> np.ndarray:
        """
        Band pass filter to select the part of the spectrum in which the event is localized.
        """
        y, zf = signal.sosfilt(self.filt_params, block, axis=1, zi=self.zi)
        self.zi = zf
        return y