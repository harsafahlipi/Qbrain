
"""
QuantumDepthwiseConv2d — Depthwise conv with optional quantum global features.
------------------------------------------------------------------------------
Performs depthwise convolution, then (optionally) augments a subset of channels
with a small quantum projection computed on global pooled features.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Optional
# endregion

# region ========== Third-Party Imports ==========
import torch
import torch.nn as nn
import torch.nn.functional as F
# endregion

# region ========== Internal ==========
from .fast_quantum_layer import FastQuantumLayer
# endregion


class QuantumDepthwiseConv2d(nn.Module):
    """Depthwise conv + optional quantum augmentation on global features."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        use_quantum: bool = True,
        quantum_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_quantum = bool(use_quantum)
        self.quantum_ratio = float(quantum_ratio)

        # Classical depthwise conv
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU6(inplace=True)

        # Quantum settings
        self.quantum_channels = max(1, int(self.in_channels * self.quantum_ratio)) if self.use_quantum else 0
        if self.quantum_channels > 0:
            self.quantum_layer = FastQuantumLayer(n_qubits=min(self.quantum_channels, 4), use_quantum=self.use_quantum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise_conv(x)
        out = self.bn(out)
        out = self.relu(out)

        if self.use_quantum and self.quantum_channels > 0:
            n, c, h, w = out.shape
            # Global average features
            global_feat = F.adaptive_avg_pool2d(out, 1).view(n, c)  # (N, C)
            q_in = global_feat[:, :max(2, self.quantum_layer.n_qubits)]  # ensure ≥2 features
            q_out = self.quantum_layer(q_in)  # (N, n_qubits)

            # Inject back into spatial map (broadcast)
            k = min(q_out.size(1), out.size(1))
            expanded = q_out[:, :k].unsqueeze(2).unsqueeze(3).expand(n, k, h, w)
            out[:, :k, :, :] = out[:, :k, :, :] + expanded

        return out
