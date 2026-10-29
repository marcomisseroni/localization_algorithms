from abc import ABC, abstractmethod

import numpy as np

class Localizer(ABC):
    """
    Base class for each localization algorithm.
    Each algorithm receives the signal in blocks as it was arriving in real time,
    after some detection algorithm.
    """

    def __init__(self, sensor_pos: np.ndarray, fs: int, c: float | None, pre_margin: int, 
                 room_dim: list[float], low_cutoff: float, high_cutoff: float) -> None:
        self.sensor_pos = sensor_pos            # Position of the sensors
        self.N = self.sensor_pos.shape[0]       # Number of sensors
        self.fs = fs                            # Sampling frequency
        self.c = c
        self.pre_margin = pre_margin
        self.room_dim = room_dim
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    @abstractmethod
    def reset(self) -> None:
        """"
        Clear the internal state (buffers and matricies) to start a new recording.
        """

    @abstractmethod
    def load_block_from_det(self, block: np.ndarray) -> np.ndarray | None:
        """
        Loads a block in which an event is detected, adapt the required parmeters and perform some check.
        """

    @abstractmethod
    def estimate(self) -> np.ndarray | None:
        """
        Compute the estimated position of the soource.
        """