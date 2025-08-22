
"""
Quick Usage example for the Cross Attention module.
Run: python -m my_package.examples.quick_usage_cross_attention
"""
from __future__ import annotations
import numpy as np

from my_package.cross_attention import CrossAttention, SimpleResizeExtractor

def main() -> None:
    extractor = SimpleResizeExtractor(size=(4, 4), reduce_dim=16)
    attn = CrossAttention(feature_extractor=extractor, shots=1024, size=(4, 4))

    # For demo we use random vectors of length 16 (4x4).
    vec_Q = np.random.rand(16).astype(np.float32)
    vec_K_list = [np.random.rand(16).astype(np.float32) for _ in range(2)]
    vec_V_list = [np.random.rand(16).astype(np.float32) for _ in range(2)]

    out_vec, weights = attn.quantum_cross_attention(vec_Q, vec_K_list, vec_V_list)
    print("Cross Attention output shape:", out_vec.shape)
    print("Attention weights:", weights)

if __name__ == "__main__":
    main()
