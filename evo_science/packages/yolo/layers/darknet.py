from .conv import Conv
from .csp import CSP
from .spp import SPP

import torch
from torch import nn


class DarkNet(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.stage1 = self._create_stage(width[0], width[1], 3, 2)
        self.stage2 = self._create_stage(width[1], width[2], 3, 2, depth[0])
        self.stage3 = self._create_stage(width[2], width[3], 3, 2, depth[1])
        self.stage4 = self._create_stage(width[3], width[4], 3, 2, depth[2])
        self.stage5 = self._create_stage(width[4], width[5], 3, 2, depth[0], use_spp=True)

    def _create_stage(self, in_channels, out_channels, kernel_size, stride, num_csp_blocks=0, use_spp=False):
        layers = [Conv(in_channels, out_channels, kernel_size, stride)]
        if num_csp_blocks > 0:
            layers.append(CSP(out_channels, out_channels, num_csp_blocks))  # type: ignore
        if use_spp:
            layers.append(SPP(out_channels, out_channels))  # type: ignore
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward through each stage and track outputs from stage 3, 4, and 5
        x = self.stage1(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)

        return p3, p4, p5
