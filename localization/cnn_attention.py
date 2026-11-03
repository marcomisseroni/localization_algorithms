import numpy as np
import torch
import math

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

class Attention(nn.Module):
    """
    Block composed by multi-head attention and add plus layer norm.
    """
    def __init__(self, att_size, num_heads):
        super().__init__()

        self.at = nn.MultiheadAttention(embed_dim=att_size, num_heads=num_heads, batch_first=True, dropout=0.05)
        self.ln = nn.LayerNorm(att_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, att_size)
        at = self.at(x, x, x)[0]        # (B, T, att_size)  self-attention, [1] are the attention weights
        x = self.ln(x + at)             # (B, T, att_size)
        return x

class SELDNet(nn.Module):
    """
    Network class used by the localizer.
    """
    def __init__(self, in_size: int):
        super().__init__()
        # (out_ch, pool, stride, kernel), output shape of each block for input (B, in_size, L)
        CONV_PARAMS = [
            (8, 5, 2, 5),       # (B, 8, ~L / 10)
            (32, 4, 2, 5),      # (B, 32, ~L / 80)
            (128, 6, 1, 5),     # (B, 128, ~L / 480)
            (128, 5, 1, 5),     # (B, 128, ~L / 2400)
        ]
        blocks = []
        ch = in_size
        for out_ch, pool, stride, kernel in CONV_PARAMS:
            blocks.append(ConvBlock(in_size=ch, out_conv_size=out_ch, pooling_size=pool, stride=stride,
                                    kernel_size=kernel, alpha=1, p_drop=0.05))
            ch = out_ch
        self.conv_blocks = nn.Sequential(*blocks)
        
        blocks = []
        for i in range(4):
            blocks.append(Attention(128, 8))
        self.att_blocks = nn.Sequential(*blocks)

        blocks = []
        blocks.append(nn.Linear(128, 128))
        blocks.append(nn.ReLU())
        blocks.append(nn.Linear(128, 3))
        self.fc_block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_size, L)
        x = self.conv_blocks(x)     # (B, 128, ~L / 2400)
        x = x.transpose(1, 2)       # (B, T, 128)  T = ~L / 2400
        x = self.att_blocks(x)      # (B, T, 128)
        x = self.fc_block(x)        # (B, T, 3)
        x1 = torch.tanh(x[:, :, 0:2])   # (B, T, 2)  activity-coupled direction (dx, dy), norm = activity
        x2 = torch.relu(x[:, :, 2:])    # (B, T, 1)  distance
        return torch.cat([x1, x2], dim=-1)  # (B, T, 3)

class CNNAttention(Localizer):
    """
    Neural network-based localization.
    The dimension of the output is (B, T, 3) where:
    B is the batch,
    T is the number of temporal frames, each one covering ~2400 input samples (~150 ms at 16 kHz),
    3 is the estimate: (dx, dy, d), in particular (dx, dy) is the direction of the source and it
    is also used as detector: if sqrt(dx**2 + dy**2) > thr an event is detected. Instead d is the distance
    between source and the center of mass of the array.
    """
    def __init__(self, sensor_pos: np.ndarray, fs: int, c: float | None, pre_margin: int, 
                 room_dim: list[float], low_cutoff: float, high_cutoff: float, det_thr: float):
        super().__init__(sensor_pos, fs, c, pre_margin, room_dim, low_cutoff, high_cutoff)
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.model = SELDNet(self.N).to(self.device)
        self.model.eval()
        self.block = None
        self.det_thr = det_thr
        self.arr_center = self.sensor_pos.mean(axis=0)[0:2]

    @override 
    def load_block_from_det(self, block: np.ndarray) -> np.ndarray | None:
        # block: (N, L)
        if(block.shape[0] != self.N or block.shape[1] < 2400):
            print("Error: wrong dimension of the block")
            return None
        block = torch.from_numpy(block).float().unsqueeze(0)    # (1, N, L)
        block = block.to(self.device)
        self.block = block                                      # (1, N, L)
        return self.estimate()

    @override
    def reset(self) -> None:
        self.block = None

    @override
    def estimate(self) -> np.ndarray | None:
        if(self.block is None):
            return None
        with torch.no_grad():
            y = self.model(self.block)
        y = y.cpu().numpy()
        estimates = []
        for b in range(y.shape[0]):
            for t in range(y.shape[1]):
                norm = math.sqrt(y[b, t, 0]**2 + y[b, t, 1]**2)
                if(norm > self.det_thr):
                    unit_vec = np.array([y[b, t, 0], y[b, t, 1]]) / norm
                    estimates.append(unit_vec * y[b, t, 2] + self.arr_center)

        est = np.array(estimates)
        if(len(est) == 0):
            est = None
        return est