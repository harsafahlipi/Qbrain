
"""
Utilities for the QTransformer Decoder block.
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


def aer_available() -> bool:
    """Return True if Aer/Estimator are available in the environment."""
    return bool(_AER_AVAILABLE and Estimator is not None and AerSimulator is not None)


def get_estimator() -> Optional[Any]:
    """Create and return an Estimator if available, else None (silent)."""
    if not aer_available():
        return None
    try:
        return Estimator()
    except Exception:
        try:
            backend = AerSimulator()
            return Estimator(backend=backend)  # type: ignore
        except Exception:
            return None


def require_positive(name: str, value: int) -> None:
    """Validate positive integer parameters."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def build_z_observables(n_qubits: int) -> List[SparsePauliOp]:
    """Return a Z-on-i (I elsewhere) observable for each qubit."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    obs: List[SparsePauliOp] = []
    for i in range(n_qubits):
        s = ['I'] * n_qubits
        s[i] = 'Z'
        obs.append(SparsePauliOp.from_list([(''.join(s), 1.0)]))
    return obs
