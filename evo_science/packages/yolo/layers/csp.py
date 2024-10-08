import torch
import torch.nn as nn

from evo_science.packages.yolo.layers.conv import Conv
from evo_science.packages.yolo.layers.residual import Residual


class CSP(nn.Module):
    """
    CSPNet: A New Backbone that can Enhance Learning Capability of CNN

    The CSP structure is defined as:
    CSP(x) = Conv(Concat(Conv1(x), Conv2(x), Residual1(Conv2(x)), ..., ResidualN(Conv2(x))))

        Where:
        - Conv1, Conv2 are convolutional layers that split the input into two branches
        - Residual1, ..., ResidualN are residual blocks applied to the right branch
        - Concat concatenates the outputs of all branches
        - Conv is a final convolutional layer to combine the features

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_residuals (int, optional): Number of residual blocks. Defaults to 1.
            use_residual_add (bool, optional): Whether to use residual addition in Residual blocks. Defaults to True.

        Reference: https://arxiv.org/abs/1911.11929
    """

    def __init__(self, in_channels, out_channels, num_residuals=1, use_residual_add=True):
        super().__init__()
        self.conv_left = Conv(in_channels, out_channels // 2)
        self.conv_right = Conv(in_channels, out_channels // 2)
        self.conv_bottom = Conv((2 + num_residuals) * out_channels // 2, out_channels)
        self.residuals = nn.ModuleList(Residual(out_channels // 2, use_residual_add) for _ in range(num_residuals))

    def forward(self, x):
        left_branch = self.conv_left(x)
        right_branch = self.conv_right(x)

        features = [left_branch, right_branch]
        for residual in self.residuals:
            features.append(residual(features[-1]))

        return self.conv_bottom(torch.cat(features, dim=1))
