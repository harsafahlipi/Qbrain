
"""
Quick Usage example for the Masked Self-Attention module.
Run: python -m my_package.examples.quick_usage_masked_self_attention
"""
from __future__ import annotations
import numpy as np

from my_package.masked_self_attention import MaskedSelfAttention, SimpleResizeExtractor

def main() -> None:
    extractor = SimpleResizeExtractor(size=(4, 4), reduce_dim=16)
    attn = MaskedSelfAttention(feature_extractor=extractor, shots=1024, size=(4, 4))

    vec_Q = np.random.rand(16).astype(np.float32)
    vec_K = np.random.rand(16).astype(np.float32)
    vec_V = np.random.rand(16).astype(np.float32)

    out_vec = attn.quantum_masked_self_attention(vec_Q, vec_K, vec_V, mask_size=16)
    print("Masked Self-Attention output shape:", out_vec.shape)

if __name__ == "__main__":
    main()
