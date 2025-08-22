
"""
Quick Usage for QTransformer Encoder.
Run: python -m my_package.examples.quick_usage_qtransformer_encoder
"""
from __future__ import annotations
import torch

from my_package.qtransformer_encoder import QuantumTransformerModel

def main() -> None:
    model = QuantumTransformerModel(num_qubits=4, num_classes=10, num_layers=1, depth=1)
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    print("Output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
