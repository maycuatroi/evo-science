import torch
import torch.nn as nn
from ..conv import Conv


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv2(self.conv1(x))
        if self.shortcut:
            x += residual
        return x
