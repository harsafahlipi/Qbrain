
"""
Quick Usage example for the QCBAM module.
Run: python -m my_package.examples.quick_usage_QCBAM
(or)  python examples/quick_usage_QCBAM.py when working locally.
"""
from __future__ import annotations
import numpy as np

# Local dev path import style:
from my_package.QCBAM import QCBAM, SimpleResizeExtractor

def main() -> None:
    extractor = SimpleResizeExtractor(size=(4, 4))
    QCBAM = QCBAM(feature_extractor=extractor)

    vec_Q = np.random.rand(16).astype(np.float32)
    vec_K = np.random.rand(16).astype(np.float32)
    vec_V = np.random.rand(16).astype(np.float32)

    out_vec = QCBAM.quantum_CBAM(vec_Q, vec_K, vec_V)
    print("Output shape:", out_vec.shape)

if __name__ == "__main__":
    main()
