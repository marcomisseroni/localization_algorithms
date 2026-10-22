import numpy as np
import json

from scipy.io import wavfile
from pathlib import Path

class Sample:
    def __init__(self, file_path: Path) -> None:
        self.data_path = file_path.parent
        self.cursor = 0

        with open(file_path, "r") as file:
            data = json.load(file)

        self.fs = data["sample_rate_hz"]
        self.source_pos = np.array(data["source_xyz_m"])
        self.N = data["microphone_count"]
        self.wav_files = data["mono_wav_files"]
        self.sensors_pos = np.array(data["microphones_xyz_m"])
        self.c = data["simulation"]["speed_of_sound"]
        self.signals = []

        for i in range(self.N):
            path = self.data_path / self.wav_files[i]
            _, x = wavfile.read(str(path))
            self.signals.append(x)
        if(x.dtype == np.int16):
            self.signals = np.stack(self.signals).astype(float) / 32768
        else:
            print("Error: data type is not int16")

    def rewind(self) -> None:
        self.cursor = 0

    def read(self, hop: int) -> np.ndarray | None:
        """
        Load some data.
        """
        if(self.cursor + hop > self.signals.shape[1]):
            return None
        block = self.signals[:, self.cursor:self.cursor + hop].copy()
        self.cursor += hop
        return block

def main():
    file_path = "/home/m4rch0/marco/master/master_thesis/acustic-array-simulator/outputs/selkd_batch_20261021_094456_763978/rep0001_file001_test_chirp.json"
    sample = Sample(file_path)

if __name__ == "__main__":
    main()