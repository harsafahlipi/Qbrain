
"""
Quantum Hierarchical Embedding (QHEB) — Feature Map
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Tuple
# endregion

# region ========== Third-Party Imports ==========
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
# endregion


# region ========== Core API ==========
def create_qheb_feature_map(num_qubits: int, reps: int = 2, *, param_prefix: str = "x") -> QuantumCircuit:
    """
    Build a Hierarchical Embedding feature map circuit.

    Structure per repetition:
      1) Per-qubit rotations: RY(x_i) then RZ(x_i)
      2) Hierarchical entanglement: CZ on (0,1), (2,3), ...
      3) Cross-feature entanglement: CZ on (1,2), (3,4), ...

    Args:
        num_qubits: Number of qubits (>= 2 recommended)
        reps: Number of hierarchical repetitions
        param_prefix: Symbol prefix for the data parameters (default: 'x')

    Returns:
        QuantumCircuit: Parameterized feature map circuit with parameters named f"{param_prefix}[i]".
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if reps <= 0:
        raise ValueError("reps must be positive")

    fm = QuantumCircuit(num_qubits)
    x = ParameterVector(param_prefix, num_qubits)

    for _ in range(reps):
        # 1) Single-qubit rotations
        for i in range(num_qubits):
            fm.ry(x[i], i)
            fm.rz(x[i], i)

        # 2) Hierarchical entanglement (pairwise even->odd)
        for i in range(0, num_qubits - 1, 2):
            fm.cz(i, i + 1)

        # 3) Cross-feature interactions (pairwise odd->next)
        for i in range(1, num_qubits - 1, 2):
            fm.cz(i, i + 1)

    return fm
# endregion


__all__ = ["create_qheb_feature_map"]
