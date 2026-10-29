import numpy as np
import json

from scipy.io import wavfile
from pathlib import Path
from scipy import signal

class Sample:
    def __init__(self, file_path: Path, low_cutoff: float, high_cutoff: float, filter_order: int) -> None:
        self.data_path = file_path.parent
        self.cursor = 0
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.filter_order = filter_order

        with open(file_path, "r") as file:
            data = json.load(file)

        self.fs = data["sample_rate_hz"]
        self.source_pos = np.array(data["source_xyz_m"])
        self.N = data["microphone_count"]
        self.wav_files = data["mono_wav_files"]
        self.sensors_pos = np.array(data["microphones_xyz_m"])
        self.c = data["simulation"]["speed_of_sound"]
        self.room_dim = data["simulation"]["room_dimensions"]
        self.filt_params = signal.butter(self.filter_order, [self.low_cutoff, self.high_cutoff], btype="bandpass", fs=self.fs, output="sos")
        self.zi = np.zeros((self.filt_params.shape[0], self.N, 2))
        
        signals = []
        for i in range(self.N):
            path = self.data_path / self.wav_files[i]
            _, x = wavfile.read(str(path))
            signals.append(x)
        self.signals: np.ndarray = np.stack(signals).astype(float)     # (N, T)
        if(x.dtype == np.int16):
            self.signals /= 32768
        else:
            print("Error: data type is not int16")

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