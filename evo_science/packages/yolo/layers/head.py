import torch
from torch import nn
from abc import ABC, abstractmethod
import numpy as np

from evo_science.packages.yolo.layers.conv import Conv
from evo_science.packages.yolo.layers.dfl import DFL
from evo_science.packages.yolo.utils import make_anchors
from evo_science.packages.yolo.yolo_config import YoloConfig


class Head(nn.Module):
    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.ch = 16  # DFL channels
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = nn.Parameter(torch.zeros(self.nl), requires_grad=False)  # strides computed during build

        self.dfl = DFL(self.ch)
        self.cls_convs, self.box_convs = self._create_conv_layers(filters)

    def _create_conv_layers(self, filters):
        """
        Create the classification and box branches of the head.
        Args:
            filters: list of filters for each detection layer
        Returns:
            tuple of two lists: classification and box branches
        """
        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))

        cls_convs = nn.ModuleList([self._create_cls_branch(x, c1) for x in filters])
        box_convs = nn.ModuleList([self._create_box_branch(x, c2) for x in filters])

        return cls_convs, box_convs

    def _create_cls_branch(self, in_channels, out_channels):
        """
        Create the classification branch of the head.
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
        Returns:
            classification branch
        """
        return nn.Sequential(
            Conv(in_channels, out_channels, 3), Conv(out_channels, out_channels, 3), nn.Conv2d(out_channels, self.nc, 1)
        )

    def _create_box_branch(self, in_channels, out_channels):
        """
        Create the box branch of the head.
        Args:
            in_channels: number of input channels
            out_channels: number of output channels
        Returns:
            box branch
        """
        return nn.Sequential(
            Conv(in_channels, out_channels, 3),
            Conv(out_channels, out_channels, 3),
            nn.Conv2d(out_channels, 4 * self.ch, 1),
        )

    def forward(self, x):
        outputs = [torch.cat((self.box_convs[i](x[i]), self.cls_convs[i](x[i])), 1) for i in range(self.nl)]

        if self.training:
            return outputs

        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(outputs, self.stride, 0.5))

        x = torch.cat([output.view(outputs[0].shape[0], self.no, -1) for output in outputs], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)

        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)

        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        for box_conv, cls_conv, stride in zip(self.box_convs, self.cls_convs, self.stride):
            box_conv[-1].bias.data[:] = 1.0  # box
            cls_conv[-1].bias.data[: self.nc] = np.log(5 / self.nc / (640 / stride) ** 2)
