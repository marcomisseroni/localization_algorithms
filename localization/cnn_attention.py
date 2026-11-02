import numpy as np
import torch

from localization.base import Localizer
from typing import override
from torch import nn

class ConvBlock(nn.Module):
    """
    Convolutional block at the beginning, repeated 4 times.
    """
    def __init__(self, in_size: int, out_conv_size: int, pooling_size: int, stride: int, kernel_size: int, alpha: float, p_drop: float):
        super().__init__()

        self.cv = nn.Conv1d(in_size, out_conv_size, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_conv_size)
        self.mp = nn.MaxPool1d(pooling_size)
        self.drop = nn.Dropout(p_drop)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(out_conv_size, int(out_conv_size * alpha))
        self.fc2 = nn.Linear(int(out_conv_size * alpha), out_conv_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_size, L)
        x = self.cv(x)                              # (B, out_conv_size, ceil(L / stride))  [kernel_size odd]
        x = torch.relu(self.bn(x))                  # (B, out_conv_size, ceil(L / stride))
        x = self.mp(x)                              # (B, out_conv_size, L')  L' = floor(ceil(L / stride) / pooling_size)
        drop = self.drop(x)                         # (B, out_conv_size, L')
        x = self.gap(drop).squeeze(-1)              # (B, out_conv_size)
        x = torch.relu(self.fc1(x))                 # (B, int(out_conv_size * alpha))
        x = torch.sigmoid(self.fc2(x))              # (B, out_conv_size)
        return drop * x.unsqueeze(-1)               # (B, out_conv_size, L')

class SELDNet(nn.Module):
    """
    Network class used by the localizer.
    """
    def __init__(self, in_size: int):
        super().__init__()
        self.conv_block1 = ConvBlock(in_size=in_size, out_conv_size=8, pooling_size=5, stride=2,
                                     kernel_size=5, alpha=1, p_drop=0.05)
        self.conv_block2 = ConvBlock(in_size=8, out_conv_size=32, pooling_size=4, stride=2,
                                     kernel_size=5, alpha=1, p_drop=0.05)
        self.conv_block3 = ConvBlock(in_size=32, out_conv_size=128, pooling_size=6, stride=1,
                                     kernel_size=5, alpha=1, p_drop=0.05)
        self.conv_block4 = ConvBlock(in_size=128, out_conv_size=128, pooling_size=5, stride=1,
                                     kernel_size=5, alpha=1, p_drop=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_size, L)
        x = self.conv_block1(x)     # (B, 8, ~L / 10)      stride 2, pool 5
        x = self.conv_block2(x)     # (B, 32, ~L / 80)     stride 2, pool 4
        x = self.conv_block3(x)     # (B, 128, ~L / 480)   stride 1, pool 6
        x = self.conv_block4(x)     # (B, 128, ~L / 2400)  stride 1, pool 5
        return x

class CNNAttention(Localizer):
    """
    Neural network-based localization.
    """
    def __init__(self, sensor_pos: np.ndarray, fs: int, c: float | None, pre_margin: int, 
                 room_dim: list[float], low_cutoff: float, high_cutoff: float):
        super().__init__(sensor_pos, fs, c, pre_margin, room_dim, low_cutoff, high_cutoff)
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = SELDNet(self.N).to(self.device)
        self.model.eval()

    @override 
    def load_block_from_det(self, block: np.ndarray) -> np.ndarray | None:
        block = torch.from_numpy(block).float().unsqueeze(0)
        block = block.to(self.device)
        with torch.no_grad():
            y = self.model(block)
        print(y.shape)

    @override
    def reset(self) -> None:
        pass

    @override
    def estimate(self) -> np.ndarray | None:
        pass