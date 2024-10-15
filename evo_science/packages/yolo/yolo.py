from evo_science.dl.abstract_torch_model import AbstractTorchModel
from evo_science.packages.yolo.layers.backbone import Backbone
from evo_science.packages.yolo.layers.conv import Conv
from evo_science.packages.yolo.layers.dark_fpn import DarkFPN
from evo_science.packages.yolo.layers.darknet import DarkNet
from evo_science.packages.yolo.layers.head import Head
from evo_science.packages.yolo.layers.neck import Neck
from evo_science.packages.yolo.losses.yolo_loss import YoloLoss
from evo_science.packages.yolo.yolo_config import YoloConfig
import torch
import torch.nn as nn


class Yolo(AbstractTorchModel):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.backbone = DarkNet(width, depth)
        self.neck = DarkFPN(width, depth)
        self.num_classes = num_classes
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self._initialize_head(width, num_classes)
        self.stride = self.head.stride

    def _initialize_head(self, width, num_classes):
        img_dummy = torch.zeros(1, 3, 256, 256)
        strides = []
        for x in self.forward(img_dummy):
            strides.append(256 / x.shape[-2])
        strides_tensor = torch.tensor(strides)
        self.head.stride = nn.Parameter(strides_tensor, requires_grad=False)
        self.head.initialize_biases()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(list(x))

    def fuse(self):
        for module in self.modules():
            if isinstance(module, Conv) and hasattr(module, "norm"):
                self._fuse_module(module)
        return self

    def _fuse_module(self, module):
        module.conv = self._fuse_conv_and_norm(module.conv, module.norm)
        module.forward = module.fuse_forward
        delattr(module, "norm")

    @staticmethod
    def _fuse_conv_and_norm(conv, norm):
        fused_conv = (
            nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
        fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
        fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

        return fused_conv

    def get_criterion(self):
        return YoloLoss(
            num_classes=self.num_classes,
            num_outputs=self.head.num_outputs,
            num_channels=self.head.num_channels,
            stride=self.stride,
        )
