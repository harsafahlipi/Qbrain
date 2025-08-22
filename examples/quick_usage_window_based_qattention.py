
"""
Quick Usage for window-based quantum self-attention.
Run: python -m my_package.examples.quick_usage_window_based_qattention
"""
from __future__ import annotations
import numpy as np
from my_package.window_based_qattention import WindowBasedQSelfAttention, WindowAttentionConfig

def main() -> None:
    cfg = WindowAttentionConfig(window_size=4, shots=512)
    model = WindowBasedQSelfAttention(cfg)
    rng = np.random.default_rng(0)
    vec_Q = rng.normal(size=16).astype("float32")
    vec_K = rng.normal(size=16).astype("float32")
    vec_V = rng.normal(size=16).astype("float32")
    out = model(vec_Q, vec_K, vec_V)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()
