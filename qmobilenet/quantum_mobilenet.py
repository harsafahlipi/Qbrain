
"""
QuantumMobileNet — A compact MobileNet-like CNN with optional quantum blocks.
"""

from __future__ import annotations

# region ========== Third-Party Imports ==========
import torch
import torch.nn as nn
# endregion


class QuantumMobileNet(nn.Module):
    """
    MobileNet-style model with QuantumMobileNetBlocks.
    Designed for small grayscale images (e.g., MNIST, 1×28×28).
    """

    def __init__(self, num_classes: int = 10, use_quantum: bool = True, quantum_ratio: float = 0.25) -> None:
        super().__init__()
        self.use_quantum = bool(use_quantum)
        self.quantum_ratio = float(quantum_ratio)

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # Reduced-depth MobileNet stages
        from .quantum_mobilenet_block import QuantumMobileNetBlock
        self.blocks = nn.Sequential(
            QuantumMobileNetBlock(32, 64, 1, self.use_quantum, self.quantum_ratio),
            QuantumMobileNetBlock(64, 128, 2, self.use_quantum, self.quantum_ratio),
            QuantumMobileNetBlock(128, 256, 2, self.use_quantum, self.quantum_ratio),
            QuantumMobileNetBlock(256, 512, 2, self.use_quantum, self.quantum_ratio),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming-style initialization for convs; standard for linear/BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
