
"""
Quick Usage for Quantum Slot Attention (qslot_attention).
Run: python -m my_package.examples.quick_usage_slot_attention
"""
from __future__ import annotations
import numpy as np
from my_package.qslot_attention import QuantumSlotAttention

def main() -> None:
    feature_dim = 16
    vec_K = np.random.rand(feature_dim).astype("float32")
    vec_V = np.random.rand(feature_dim).astype("float32")

    model = QuantumSlotAttention(feature_dim=feature_dim)
    out = model(vec_K, vec_V)
    print("Output shape:", out.shape)  # (num_slots * feature_dim,)

if __name__ == "__main__":
    main()
