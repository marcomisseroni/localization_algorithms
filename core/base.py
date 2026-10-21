from abc import ABC, abstractmethod

import numpy as np

class Localizer(ABC):
    """
    Base class for each localization algorithm.
    Each algorithm receives the signal in blocks as it was arriving in real time,
    after some detection algorithm.
    """

    def __init__(self, sensor_pos: np.ndarray, fs: int, c: float) -> None:
        self.sensor_pos = sensor_pos            # Position of the sensors
        self.N = self.sensor_pos.shape[0]       # Number of sensors
        self.fs = fs                            # Sampling frequency
        self.c = c

    @abstractmethod
    def push(self, block: np.ndarray) -> np.ndarray | None:
        """
        block: (N, hop) new samples in chronological order (last column = most recent).
        Returns the estimated source position (3,) or None if no estimate is available yet.
        """
    @abstractmethod
    def reset(self) -> None:
        """"
        Clear the internal state (buffers and matricies) to start a new recording.
        """