import torch
import torch.nn as nn

from evo_science.packages.yolo.layers.conv import Conv
from evo_science.packages.yolo.layers.residual import Residual


class CSP(nn.Module):
    def __init__(self, in_channels, out_channels, num_residuals=1, use_residual_add=True):
        super().__init__()
        self.branch1 = Conv(in_channels, out_channels // 2)
        self.branch2 = nn.Sequential(
            Conv(in_channels, out_channels // 2),
            *[Residual(out_channels // 2, use_residual_add) for _ in range(num_residuals)]
        )
        self.final_conv = Conv((2 + num_residuals) * out_channels // 2, out_channels)

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        combined = torch.cat([branch1_output, branch2_output], dim=1)
        return self.final_conv(combined)
