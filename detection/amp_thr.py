import numpy as np

from scipy.ndimage import uniform_filter1d
from detection.base import Detector
from typing import override

class AmpThr(Detector):
    """
    Simple threshold detector.
    """

    def __init__(self, pre_margin: int, post_margin: int, fs: int, N: int, 
                 min_thr: float, window: int, learning_rate: float, k:float) -> None:
        super().__init__(pre_margin=pre_margin, post_margin=post_margin, fs=fs, N=N)
        self.thr = np.zeros(self.N)
        self.window = window
        self.pre_buffer = np.zeros((self.N, self.pre_margin))
        self.block = None
        self.rec = False
        self.event_buffer = np.zeros((self.N, 0))
        self.prev_mask_end = 0
        self.learning_rate = learning_rate
        self.k = k
        self.first_block = True
        self.min_thr = min_thr

    @override
    def detect(self, block: np.ndarray) -> list[np.ndarray]:
        if(self.block is not None):
            self.pre_buffer = self.block[:, -self.pre_margin:]
        self.block = block
        events = []
        block_zm = block - np.mean(block, axis=1, keepdims=True)
        mov_avg = uniform_filter1d(np.abs(block_zm), size=self.window, axis = 1) # Compute the moving average on each row
        self.thr_update(mov_avg)
        mask = mov_avg > self.thr[:, None]   # Apply the threshold
        event_mask = mask.any(axis=0)   # Compute the or col-wise
        d = np.diff(event_mask.astype(np.int8), prepend=self.prev_mask_end)    # Compute the difference between consecutive elements (converts bool -> int and add a 0 at the beginning and at the end)
        starts = np.nonzero(d == 1)[0]  # Extract the indexes of the 1 -> in diff 1-0 = 1 (beginning of the event)
        ends = np.nonzero(d == -1)[0]   # Extract the indexes of the -1 -> in diff 0-1=1 (end of the event)
        self.prev_mask_end = event_mask[-1]
        if(self.rec and len(ends) > 0):
            self.event_buffer = np.concatenate((self.event_buffer, block[:, 0:ends[0]]), axis=1)
            events.append(self.event_buffer) 
            ends = ends[1:]
            self.rec = False
            self.event_buffer = np.zeros((self.N, 0))
        elif(self.rec and len(ends) == 0):
            self.rec = True
            self.event_buffer = np.concatenate((self.event_buffer, block), axis=1)
            return events
        if(event_mask[-1] == True and len(starts) > 0):
            self.rec = True
            self.event_buffer = np.concatenate((self.add_pre_margin(starts[-1]), block[:, starts[-1]:]), axis=1)
            starts = starts[:-1]
        while(len(ends) > 0):
            events.append(np.concatenate((self.add_pre_margin(starts[0]), block[:, starts[0]:ends[0]]), axis=1))
            starts = starts[1:]
            ends = ends[1:]
        return events

    @override
    def reset(self) -> None:
        self.pre_buffer = np.zeros((self.N, self.pre_margin))
        self.rec = False
        self.event_buffer = np.zeros((self.N, 0))
        self.prev_mask_end = 0
        self.block = None
        self.thr = np.zeros(self.N)
        self.first_block = True

    def add_pre_margin(self, start_index: int) -> np.ndarray:
        """
        Return the part of the pre margin to stack in front of current block.
        Empty if it's not needed.
        """
        if(start_index < self.pre_margin):
            event = np.concatenate((self.pre_buffer[:, start_index-self.pre_margin:], self.block[:, 0: start_index]), axis=1)
        else:
            event = self.block[:, start_index-self.pre_margin:start_index]
        return event

    def thr_update(self, mov_avg: np.ndarray) -> None:
        """
        Dynamically update the threshold to adapt for different noise levels.
        """
        if(self.first_block):
            self.first_block = False
            self.thr = np.maximum(self.k * np.median(mov_avg, axis=1), self.min_thr)
        noise = np.maximum(np.median(mov_avg, axis=1) * self.k, self.min_thr)
        self.thr = (1 - self.learning_rate) * self.thr + self.learning_rate * noise 