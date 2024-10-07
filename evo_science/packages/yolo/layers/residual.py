import torch.nn as nn

from evo_science.packages.yolo.layers.conv import Conv


class Residual(nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = nn.Sequential(Conv(ch, ch, 3), Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)
