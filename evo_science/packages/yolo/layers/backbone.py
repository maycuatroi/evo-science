import torch.nn as nn

from evo_science.packages.yolo.layers.c2f.c2f import C2f
from evo_science.packages.yolo.layers.conv import Conv
from evo_science.packages.yolo.layers.sppf import SPPF
from evo_science.packages.yolo.yolo_config import YoloConfig


class Backbone(nn.Module):
    def __init__(self, config: YoloConfig, in_channels=3, shortcut=True):
        super().__init__()
        d, w, r = config.get_scaling_factors()

        self.layers = nn.ModuleList(
            [
                self._create_block(in_channels, int(64 * w), 3, 2, 1),
                self._create_block(int(64 * w), int(128 * w), 3, 2, 1),
                C2f(int(128 * w), int(128 * w), num_bottlenecks=int(3 * d), shortcut=shortcut),
                self._create_block(int(128 * w), int(256 * w), 3, 2, 1),
                C2f(int(256 * w), int(256 * w), num_bottlenecks=int(6 * d), shortcut=shortcut),
                self._create_block(int(256 * w), int(512 * w), 3, 2, 1),
                C2f(int(512 * w), int(512 * w), num_bottlenecks=int(6 * d), shortcut=shortcut),
                self._create_block(int(512 * w), int(512 * w * r), 3, 2, 1),
                C2f(int(512 * w * r), int(512 * w * r), num_bottlenecks=int(3 * d), shortcut=shortcut),
                SPPF(int(512 * w * r), int(512 * w * r)),
            ]
        )

    def _create_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in [4, 6, 9]:  # Indices for out1, out2, and out3
                outputs.append(x)
        return tuple(outputs)
