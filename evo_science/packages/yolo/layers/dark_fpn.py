import torch
import torch.nn as nn
from .csp import CSP
from .conv import Conv


class DarkFPN(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.layers = nn.ModuleDict(
            {
                "csp_0": CSP(width[4] + width[5], width[4], depth[0], False),
                "csp_1": CSP(width[3] + width[4], width[3], depth[0], False),
                "csp_2": CSP(width[3] + width[4], width[4], depth[0], False),
                "csp_3": CSP(width[4] + width[5], width[5], depth[0], False),
                "conv_0": Conv(width[3], width[3], 3, 2),
                "conv_1": Conv(width[4], width[4], 3, 2),
            }
        )

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tuple of three tensors, representing the output of the last three layers of the backbone
        Returns:
            tuple of three tensors, representing the output of the FPN
        """
        p3, p4, p5 = x

        # Top-down path
        p4 = self.layers["csp_0"](torch.cat([self.upsample(p5), p4], 1))
        p3 = self.layers["csp_1"](torch.cat([self.upsample(p4), p3], 1))

        # Bottom-up path
        p4 = self.layers["csp_2"](torch.cat([self.layers["conv_0"](p3), p4], 1))
        p5 = self.layers["csp_3"](torch.cat([self.layers["conv_1"](p4), p5], 1))

        return p3, p4, p5
