from abc import ABC, abstractmethod

import numpy as np

class Detector(ABC):
    """
    Base class for each detection algorithm.
    """
    def __init__(self, pre_margin: int, post_margin: int, fs: int, N: int) -> None:
        self.pre_margin = pre_margin
        self.post_margin = post_margin
        self.fs = fs
        self.N = N

    @abstractmethod
    def detect(self, block: np.ndarray) -> list:
        """
        Method that returns a list of sub-blocks containing events.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Used after each event to reset the state.
        """