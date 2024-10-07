import torch
import torch.nn as nn

from evo_science.packages.yolo.layers.c2f.c2f import C2f
from evo_science.packages.yolo.layers.conv import Conv
from evo_science.packages.yolo.layers.upsample import Upsample
from evo_science.packages.yolo.yolo_config import YoloConfig


class Neck(nn.Module):
    def __init__(self, config: YoloConfig):
        super().__init__()
        d, w, r = config.get_scaling_factors()

        self.up = Upsample()  # no trainable parameters
        self.c2f_layers = nn.ModuleList(
            [
                C2f(
                    in_channels=int(512 * w * (1 + r)),
                    out_channels=int(512 * w),
                    num_bottlenecks=int(3 * d),
                    shortcut=False,
                ),
                C2f(in_channels=int(768 * w), out_channels=int(256 * w), num_bottlenecks=int(3 * d), shortcut=False),
                C2f(in_channels=int(768 * w), out_channels=int(512 * w), num_bottlenecks=int(3 * d), shortcut=False),
                C2f(
                    in_channels=int(512 * w * (1 + r)),
                    out_channels=int(512 * w * r),
                    num_bottlenecks=int(3 * d),
                    shortcut=False,
                ),
            ]
        )

        self.cv_layers = nn.ModuleList(
            [
                Conv(in_channels=int(256 * w), out_channels=int(256 * w), kernel_size=3, stride=2, padding=1),
                Conv(in_channels=int(512 * w), out_channels=int(512 * w), kernel_size=3, stride=2, padding=1),
            ]
        )

    def forward(self, x_res_1, x_res_2, x):
        # x_res_1, x_res_2, x = output of backbone
        res_1 = x  # for residual connection

        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)

        res_2 = self.c2f_layers[0](x)  # for residual connection

        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)

        out_1 = self.c2f_layers[1](x)

        x = self.cv_layers[0](out_1)

        x = torch.cat([x, res_2], dim=1)
        out_2 = self.c2f_layers[2](x)

        x = self.cv_layers[1](out_2)

        x = torch.cat([x, res_1], dim=1)
        out_3 = self.c2f_layers[3](x)

        return out_1, out_2, out_3
