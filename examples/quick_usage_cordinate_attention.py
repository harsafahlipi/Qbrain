
"""
Quick Usage example for the Coordinate Attention module.
Run: python -m my_package.examples.quick_usage_cordinate_attention
"""
from __future__ import annotations
import numpy as np

from my_package.cordinate_attention import CoordinateAttention, SimpleResizeExtractor

def main() -> None:
    extractor = SimpleResizeExtractor(size=(4, 4), reduce_dim=16)
    attn = CoordinateAttention(feature_extractor=extractor, shots=1024, size=(4, 4))

    # For demo purpose we use random vectors of length 16 (4x4).
    vec_Q = np.random.rand(16).astype(np.float32)
    vec_K = np.random.rand(16).astype(np.float32)
    vec_V = np.random.rand(16).astype(np.float32)

    out = attn.quantum_cordinate_attention(vec_Q, vec_K, vec_V)
    print("Coordinate Attention output shape:", out.shape)  # Expect (8,) for 4x4 -> (4+4)

if __name__ == "__main__":
    main()
