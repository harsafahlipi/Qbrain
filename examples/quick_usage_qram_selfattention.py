
"""
Quick Usage example for QRAM Self-Attention.
Run: python -m my_package.examples.quick_usage_qram_selfattention
"""
from __future__ import annotations
import torch

from my_package.qram_selfattention import QRAM_QSANN_Working

def main() -> None:
    model = QRAM_QSANN_Working(embed_dim=32, num_layers=2, num_heads=1, qram_size=8)
    # Fake sequence: (batch=2, seq_len=10, embed_dim=32)
    x = torch.randn(2, 10, 32)
    y = model(x)
    print("Output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
