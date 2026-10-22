import numpy as np

from detection.base import Detector
from typing import override

class AmpThr(Detector):
    """
    Simple threshold detector.
    """

    def __init__(self, pre_margin: int, post_margin: int, fs: int, N: int, max_duration: int, thr: float) -> None:
        super().__init__(pre_margin, post_margin, fs, N, max_duration)
        self.thr = thr

    @override
    def detect(self, block: np.ndarray) -> np.ndarray | None:
        pass

    @override
    def reset(self) -> None:
        pass