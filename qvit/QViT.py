
"""
QViT — Aggregator Module
========================

Re-exports:
- PatchPreprocessor (per-sample PCA+scaler utilities)
- QViTPatchFeatureMap (CLS + per-patch ZZFeatureMap)
- create_ansatz (RealAmplitudes factory)
- create_vqc (optional, if qiskit-ml available)
"""

from __future__ import annotations

from .preprocess import PatchPreprocessor
from .feature_map import QViTPatchFeatureMap
from .ansatz import create_ansatz

# Optional VQC hook
try:
    from .trainer import create_vqc  # type: ignore
except Exception:  # pragma: no cover
    create_vqc = None  # type: ignore

__all__ = ["PatchPreprocessor", "QViTPatchFeatureMap", "create_ansatz", "create_vqc"]
