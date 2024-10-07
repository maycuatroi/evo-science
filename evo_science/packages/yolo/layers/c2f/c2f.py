import torch
import torch.nn as nn

from evo_science.packages.yolo.layers.c2f.bottle_neck import Bottleneck
from evo_science.packages.yolo.layers.conv import Conv


class C2f(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_bottlenecks: int, shortcut: bool = True):
        super().__init__()

        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks

        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.bottlenecks = nn.ModuleList(
            [Bottleneck(self.mid_channels, self.mid_channels, shortcut) for _ in range(num_bottlenecks)]
        )

        self.conv2 = Conv((num_bottlenecks + 2) * self.mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x1, x2 = torch.split(x, self.mid_channels, dim=1)

        outputs = [x1, x2]
        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
            outputs.insert(0, x1)

        return self.conv2(torch.cat(outputs, dim=1))


if __name__ == "__main__":
    c2f = C2f(in_channels=64, out_channels=128, num_bottlenecks=2)
    print(f"{sum(p.numel() for p in c2f.parameters())/1e6} million parameters")

    dummy_input = torch.rand((1, 64, 244, 244))
    dummy_input = c2f(dummy_input)
    print("Output shape: ", dummy_input.shape)
