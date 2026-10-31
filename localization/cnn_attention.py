import numpy as np
import torch

from localization.base import Localizer
from typing import override
from torch import nn

class CNNAttention(Localizer):
    """
    Neural network-based localization.
    """
    def __init__(self, sensor_pos, fs, c, pre_margin, room_dim, low_cutoff, high_cutoff):
        super().__init__(sensor_pos, fs, c, pre_margin, room_dim, low_cutoff, high_cutoff)
        self.nn = NeuralNetwork()

    @override 
    def load_block_from_det(self, block: np.ndarray) -> np.ndarray | None:
        pass

    @override
    def reset(self) -> None:
        pass

    @override
    def estimate(self) -> np.ndarray | None:
        pass

class NeuralNetwork(nn.Module):
    """
    Network class used by the localizer.
    """
    def __init__(self):
        super().__init__()
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")