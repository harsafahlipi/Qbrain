
"""
Quick Usage example for the QFFN module.
Run: python -m my_package.examples.quick_usage_qffn
"""
from __future__ import annotations
import torch
from my_package.qffn import OptimizedQuantumFFN

def main() -> None:
    model = OptimizedQuantumFFN(input_dim=784, hidden_dim=32, output_dim=10, use_quantum=True, quantum_ratio=0.25)
    x = torch.randn(8, 784)
    y = model(x)
    print("Output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
