
"""
Utilities for the QTransformer Encoder block.
"""

from __future__ import annotations
from typing import Optional, Any, List

# Robust Estimator/Aer detection across Qiskit versions
try:
    from qiskit_aer.primitives import Estimator  # type: ignore
    from qiskit_aer import AerSimulator  # type: ignore
    _AER_AVAILABLE = True
except Exception:  # pragma: no cover
    try:
        from qiskit.primitives import Estimator  # type: ignore
        from qiskit_aer import AerSimulator  # type: ignore
        _AER_AVAILABLE = True
    except Exception:  # pragma: no cover
        try:
            from qiskit.providers.aer import AerSimulator  # type: ignore
            Estimator = None  # type: ignore
            _AER_AVAILABLE = False
        except Exception:  # pragma: no cover
            Estimator = None  # type: ignore
            AerSimulator = None  # type: ignore
            _AER_AVAILABLE = False

from qiskit.quantum_info import SparsePauliOp


def get_estimator() -> Optional[Any]:
    """Create and return an Estimator if available, else None (silent)."""
    if Estimator is None or AerSimulator is None:
        return None
    try:
        return Estimator()
    except Exception:
        try:
            backend = AerSimulator()
            return Estimator(backend=backend)  # type: ignore
        except Exception:
            return None


def build_z_observables(n_qubits: int) -> List[SparsePauliOp]:
    """Return [Z_i] observables for each qubit i ∈ [0..n_qubits-1]."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    obs: List[SparsePauliOp] = []
    for i in range(n_qubits):
        s = ['I'] * n_qubits
        s[i] = 'Z'
        obs.append(SparsePauliOp.from_list([(''.join(s), 1.0)]))
    return obs


def build_z0_observable(n_qubits: int) -> SparsePauliOp:
    """Return Z ⊗ I ⊗ ... observable (Z on qubit 0)."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    s = ['I'] * n_qubits
    s[0] = 'Z'
    return SparsePauliOp.from_list([(''.join(s), 1.0)])
