
"""
Quick Usage for qConv.
Run: python -m my_package.examples.quick_usage_qconv
"""
from __future__ import annotations
import numpy as np
from my_package.qconv import QConvAttention, quantum_conv_patch_block_amp

def main() -> None:
    rng = np.random.default_rng(0)
    patches = [
        rng.normal(size=(1,)).astype("float32"),
        rng.normal(size=(9,)).astype("float32"),
        rng.normal(size=(25,)).astype("float32"),
    ]
    # Single-patch functional API
    w0 = np.array([0.1])  # will repeat over 1 qubit (actually 1 due to min qubits)
    res0 = quantum_conv_patch_block_amp(patches[0], w0)

    # Multi-patch
    model = QConvAttention(seed=123)
    out = model(patches)  # random default weights per patch
    print("Single-patch output length:", len(res0))
    print("Concat output length:", len(out))

if __name__ == "__main__":
    main()
