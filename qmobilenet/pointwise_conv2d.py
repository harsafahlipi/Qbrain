
"""
PointwiseConv2d — standard 1x1 convolution with BN + ReLU6.
"""

from __future__ import annotations
import torch.nn as nn


class PointwiseConv2d(nn.Module):
    """A standard MobileNet pointwise (1x1) convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
