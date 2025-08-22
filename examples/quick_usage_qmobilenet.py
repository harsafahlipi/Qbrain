
"""
Quick Usage example for the QMobileNet module.
Run: python -m my_package.examples.quick_usage_qmobilenet
"""
from __future__ import annotations
import torch

from my_package.qmobilenet import QuantumMobileNet

def main() -> None:
    model = QuantumMobileNet(num_classes=10, use_quantum=True, quantum_ratio=0.25)
    x = torch.randn(2, 1, 28, 28)  # batch of 2 MNIST-like images
    y = model(x)
    print("Output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
