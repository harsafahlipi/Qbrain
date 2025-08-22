
"""
QHEB (Quantum Hierarchical Embedding Block) — Core
=================================================

Builds a hierarchical feature map with three stages per repetition:
1) Per-qubit RY/RZ rotations (feature injection)
2) Hierarchical entanglement with CZ on even pairs
3) Cross-feature CZ on odd pairs

Design principles:
- English docstrings, type hints, DRY/KISS/SOLID
- No side effects (no prints)
- Region markers to separate concerns
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Tuple
# endregion

# region ========== Third-Party Imports ==========
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
# endregion

# region ========== Internal Utilities / Config ==========
from .utils import DEFAULT_REPS, require_positive
# endregion


class QuantumHierarchicalEmbeddingBlock:
    """
    Construct a Quantum Hierarchical Embedding feature map.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the block.
    reps : int, default=DEFAULT_REPS
        Repetitions of the three-stage pattern.
    param_prefix : str, default="x"
        Prefix used for the ParameterVector.
    """

    def __init__(self, num_qubits: int, reps: int = DEFAULT_REPS, param_prefix: str = "x") -> None:
        require_positive("num_qubits", num_qubits)
        require_positive("reps", reps)
        self.num_qubits = num_qubits
        self.reps = reps
        self.param_prefix = param_prefix

    # region ---- Builder ----
    def build(self) -> Tuple[ParameterVector, QuantumCircuit]:
        """
        Build and return (parameters, feature_map_circuit).

        Returns
        -------
        (ParameterVector, QuantumCircuit)
        """
        num_qubits = self.num_qubits
        reps = self.reps
        feature_map = QuantumCircuit(num_qubits)
        params = ParameterVector(self.param_prefix, num_qubits)

        for _ in range(reps):
            # 1) Per-qubit rotations
            for i in range(num_qubits):
                feature_map.ry(params[i], i)
                feature_map.rz(params[i], i)

            # 2) Hierarchical entanglement (even pairs)
            for i in range(0, num_qubits - 1, 2):
                feature_map.cz(i, i + 1)

            # 3) Cross-feature interactions (odd pairs)
            for i in range(1, num_qubits - 1, 2):
                feature_map.cz(i, i + 1)

        return params, feature_map
    # endregion


def create_qheb_feature_map(num_qubits: int, reps: int = DEFAULT_REPS) -> QuantumCircuit:
    """
    Convenience function that returns only the feature-map circuit,
    matching the original API shape.
    """
    block = QuantumHierarchicalEmbeddingBlock(num_qubits=num_qubits, reps=reps)
    _, circuit = block.build()
    return circuit
