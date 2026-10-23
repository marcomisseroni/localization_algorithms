import numpy as np

from scipy.ndimage import uniform_filter1d
from detection.base import Detector
from typing import override

class AmpThr(Detector):
    """
    Simple threshold detector.
    """

    def __init__(self, pre_margin: int, post_margin: int, fs: int, N: int, max_duration: int, thr: float, window: int) -> None:
        super().__init__(pre_margin, post_margin, fs, N, max_duration)
        self.thr = thr
        self.window = window
        self.pre_buffer = np.zeros((self.N, self.pre_margin))
        self.block = None

    @override
    def detect(self, block: np.ndarray) -> np.ndarray | None:
        if(self.block is not None):
            self.pre_buffer = self.block[:, self.pre_margin:]
        self.block = block
        block_zm = block - np.mean(block, axis=1, keepdims=True)
        mov_avg = uniform_filter1d(np.abs(block_zm), size=self.window, axis = 1) # Compute the moving average on each row
        mask = mov_avg > self.thr   # Apply the threshold
        event_mask = mask.any(axis=0)   # Compute the and col-wise
        d = np.diff(event_mask.astype(np.int8), prepend=0, append=0)    # Compute the difference between consecutive elements (converts bool -> int and add a 0 at the beginning and at the end)
        starts = np.nonzero(d == 1)[0]  # Extract the indexes of the 1 -> in diff 1-0 = 1 (beginning of the event)
        ends = np.nonzero(d == -1)[0]   # Extract the indexes of the -1 -> in diff 0-1=1 (end of the event)
        if(len(starts) > 0):
            k = np.argmax(ends- starts) # Find the index of the event with max duration
            a = starts[k] - self.pre_margin # Take the max between 0 and the index at the beginning of the longest event - pre margin
            b = min(ends[k] + self.post_margin, block.shape[1]) # Take min between the end of the longest event + post margin and length of the block
            if(a < 0):
                block = np.concatenate((self.pre_buffer[:, a:], block[:, :b]), axis=1)
            else:
                block = block[:, a:b]
        else:
            block = None
        return block

    @override
    def reset(self) -> None:
        self.pre_buffer = np.zeros((self.N, self.pre_margin))
        self.post_buffer = np.zeros((self.N, self.post_margin))