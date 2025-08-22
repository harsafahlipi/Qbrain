
"""
QuantumMobileNetBlock — Depthwise (quantum) + Pointwise block.
"""

from __future__ import annotations
import torch.nn as nn

from .quantum_depthwise_conv2d import QuantumDepthwiseConv2d
from .pointwise_conv2d import PointwiseConv2d


class QuantumMobileNetBlock(nn.Module):
    """A single MobileNet block: depthwise (quantum) followed by pointwise conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_quantum: bool = True,
        quantum_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.depthwise = QuantumDepthwiseConv2d(
            in_channels, kernel_size=3, stride=stride, padding=1,
            use_quantum=use_quantum, quantum_ratio=quantum_ratio
        )
        self.pointwise = PointwiseConv2d(in_channels, out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
