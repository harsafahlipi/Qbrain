
"""
Utilities and configuration for the Quantum Layer Normalization module.
"""

from __future__ import annotations
from typing import List
import numpy as np

from qiskit.quantum_info import SparsePauliOp

# ---- Module-level configuration ("globals") ----
DEFAULT_LAYERS: int = 1


def require_positive(name: str, value: int) -> None:
    """Validate that an integer parameter is positive (> 0)."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def build_z_observables(n_qubits: int) -> List[SparsePauliOp]:
    """Create Z observables (Z on i, I elsewhere) for each qubit."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    observables: List[SparsePauliOp] = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        observables.append(SparsePauliOp.from_list([(''.join(pauli), 1.0)]))
    return observables
