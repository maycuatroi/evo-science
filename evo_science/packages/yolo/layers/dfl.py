import torch
import torch.nn as nn


class DFL(nn.Module):
    """
    Generalized Distribution for Object Detection, https://arxiv.org/abs/2105.04714
    """

    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)

        init_weights = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(init_weights)

    def forward(self, x):
        batch_size, channels, anchor_points = x.shape
        assert channels == 4 * self.ch, f"Expected input channels to be {4*self.ch}, but got {channels}"

        x = x.view(batch_size, 4, self.ch, anchor_points).transpose(1, 2)
        x = x.softmax(dim=1)
        x = self.conv(x)

        return x.view(batch_size, 4, anchor_points)
