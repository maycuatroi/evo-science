import torch.nn as nn
import torch.nn.functional as F
import torch


class Upsample(nn.Module):
    def __init__(self, scale_factor: int = 2, mode: str = "nearest"):
        """
        Upsample layer
        Args:
            scale_factor: int
            mode (str): "nearest" or "bilinear"
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
