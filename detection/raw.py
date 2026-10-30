import numpy as np

from detection.base import Detector
from typing import override

class Raw(Detector):
    def __init__(self, pre_margin: int, fs: int, N: int, input_size: int, stride: int, post_margin: int = 0) -> None:
        super().__init__(pre_margin=pre_margin, post_margin=post_margin, fs=fs, N=N)
        self.input_size = input_size
        self.stride = stride
        self.block = None

    @override
    def detect(self, block: np.ndarray) -> list[np.ndarray]:
        """
        Just pass a block with input_size samples per sensor.
        Used to have constant dimension input block for neural network.
        """
        if(self.block is None):
            self.block = block
        else:
            self.block = np.concatenate([self.block, block], axis=1)
        input_block_list = []
        while(True):
            input_block = self.extract_in()
            if(input_block is None):
                break
            input_block_list.append(input_block)
        return input_block_list

    @override
    def reset(self) -> None:
        self.block = None

    def extract_in(self) -> np.ndarray | None:
        """
        Checks if the attribute self.block contains enough samples to send an input block
        to a neural network.
        """
        input_block = None
        if(self.block.shape[1] >= self.input_size):
            input_block = self.block[:, 0:self.input_size]
            self.block = self.block[:, self.stride:]
        return input_block