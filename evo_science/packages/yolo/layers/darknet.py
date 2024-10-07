from .conv import Conv
from .csp import CSP
from .spp import SPP

import torch
from torch import nn


class DarkNet(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "stage1": self._create_stage(width[0], width[1], 3, 2),
                "stage2": self._create_stage(width[1], width[2], 3, 2, depth[0]),
                "stage3": self._create_stage(width[2], width[3], 3, 2, depth[1]),
                "stage4": self._create_stage(width[3], width[4], 3, 2, depth[2]),
                "stage5": self._create_stage(width[4], width[5], 3, 2, depth[0], use_spp=True),
            }
        )

    def _create_stage(self, in_channels, out_channels, kernel_size, stride, num_csp_blocks=0, use_spp=False):
        layers = [Conv(in_channels, out_channels, kernel_size, stride)]
        if num_csp_blocks > 0:
            layers.append(CSP(out_channels, out_channels, num_csp_blocks))  # type: ignore
        if use_spp:
            layers.append(SPP(out_channels, out_channels))  # type: ignore
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for i, (stage_id, layer) in enumerate(self.layers.items()):
            x = layer(x)
            if i >= 2:
                outputs.append(x)
        return outputs
