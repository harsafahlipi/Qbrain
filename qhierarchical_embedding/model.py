
"""
QHEB model helpers — optional VQC factory (when qiskit-machine-learning is available).
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Any, Optional, Callable
# endregion

# region ========== Third-Party Imports ==========
from qiskit import QuantumCircuit
# Optional: VQC imports are performed lazily in `build_vqc` for version robustness.
# endregion


# region ========== Factories ==========
def build_vqc(
    feature_map: QuantumCircuit,
    ansatz: QuantumCircuit,
    *,
    optimizer: Optional[Any] = None,
    sampler: Optional[Any] = None,
    callback: Optional[Callable[..., None]] = None,
) -> Any:
    """
    Create a VQC classifier with the given feature map and ansatz.

    Notes:
        - This function requires qiskit-machine-learning. It purposefully avoids
          side effects (no prints). If the import path changes across versions,
          it tries a few known locations and raises a RuntimeError otherwise.

    Args:
        feature_map: Parameterized data-encoding circuit
        ansatz: Variational circuit
        optimizer: Qiskit optimizer instance (e.g., COBYLA)
        sampler: Qiskit Sampler primitive
        callback: Optional callback for the training loop

    Returns:
        VQC instance

    Raises:
        RuntimeError: If a compatible VQC cannot be imported.
    """
    # Try multiple import locations to maximize compatibility
    last_err: Optional[Exception] = None
    VQC = None
    try:
        from qiskit_machine_learning.algorithms import VQC as _VQC  # type: ignore
        VQC = _VQC
    except Exception as e:
        last_err = e
        try:
            from qiskit_machine_learning.algorithms.classifiers import VQC as _VQC  # type: ignore
            VQC = _VQC
        except Exception as e2:
            last_err = e2

    if VQC is None:
        raise RuntimeError(f"VQC import failed; ensure qiskit-machine-learning is installed. Last error: {last_err}")

    # Construct VQC with provided components
    kwargs = dict(feature_map=feature_map, ansatz=ansatz)
    if optimizer is not None:
        kwargs["optimizer"] = optimizer
    if sampler is not None:
        kwargs["sampler"] = sampler
    if callback is not None:
        kwargs["callback"] = callback

    return VQC(**kwargs)
# endregion


__all__ = ["build_vqc"]
