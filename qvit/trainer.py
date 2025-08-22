
"""
Optional VQC wiring for QViT patch embeddings.
Core code stays agnostic; this module provides a small convenience factory.
"""

from __future__ import annotations
from typing import Any, Callable, Optional

# Optional imports (handled defensively)
try:
    from qiskit_machine_learning.algorithms.classifiers import VQC  # type: ignore
except Exception:  # pragma: no cover
    VQC = None  # type: ignore

try:
    from qiskit.primitives import Sampler  # type: ignore
except Exception:  # pragma: no cover
    Sampler = None  # type: ignore

try:
    # SPSA optimizer location varies across versions
    from qiskit_algorithms.optimizers import SPSA  # type: ignore
except Exception:  # pragma: no cover
    try:
        from qiskit_optimization.algorithms import SPSA  # type: ignore
    except Exception:  # pragma: no cover
        SPSA = None  # type: ignore


def create_vqc(feature_map, ansatz, optimizer: Optional[Any] = None, sampler: Optional[Any] = None, callback: Optional[Callable] = None) -> Any:
    """
    Create a VQC classifier with provided feature_map/ansatz.
    Raises RuntimeError if VQC or Sampler is unavailable.
    """
    if VQC is None or Sampler is None:
        raise RuntimeError("VQC or Sampler is not available in this environment.")
    opt = optimizer if optimizer is not None else (SPSA(maxiter=100) if SPSA is not None else None)
    smp = sampler if sampler is not None else Sampler()
    return VQC(feature_map=feature_map, ansatz=ansatz, optimizer=opt, sampler=smp, callback=callback)
