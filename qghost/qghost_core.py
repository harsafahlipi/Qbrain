
"""
QGhost (Quantum Ghost Module Block) — Core
=========================================

Builds a "ghost" quantum block by composing a feature map (ZZFeatureMap)
with either a RealAmplitudes ansatz (parameterized) or a lightweight
fixed-parameter "ghost" entangling pattern.

Design goals:
- English docstrings, type hints, KISS
- Small, testable class with no side effects
- Region markers to separate concerns
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Literal, Tuple
# endregion

# region ========== Third-Party Imports ==========
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
import numpy as np
# endregion

# region ========== Internal Utilities / Config ==========
from .utils import DEFAULT_FEATURE_MAP_REPS, DEFAULT_ANSATZ_REPS, require_positive
# endregion


class QuantumGhostBlock:
    """
    Construct a Quantum Ghost block: FeatureMap ∘ Ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the block.
    feature_map_reps : int, default=DEFAULT_FEATURE_MAP_REPS
        Repetitions for the ZZFeatureMap.
    ansatz_reps : int, default=DEFAULT_ANSATZ_REPS
        Repetitions for the ansatz pattern.
    mode : {"ghost", "real_amplitudes"}, default="ghost"
        Which ansatz to use:
        - "ghost": a simple fixed-angle RY + CNOT pattern.
        - "real_amplitudes": Qiskit's RealAmplitudes template.
    """

    def __init__(
        self,
        num_qubits: int,
        feature_map_reps: int = DEFAULT_FEATURE_MAP_REPS,
        ansatz_reps: int = DEFAULT_ANSATZ_REPS,
        mode: Literal["ghost", "real_amplitudes"] = "ghost",
    ) -> None:
        require_positive("num_qubits", num_qubits)
        require_positive("feature_map_reps", feature_map_reps)
        require_positive("ansatz_reps", ansatz_reps)

        self.num_qubits = num_qubits
        self.feature_map_reps = feature_map_reps
        self.ansatz_reps = ansatz_reps
        self.mode = mode

    # region ---- Builders ----
    def _build_feature_map(self) -> ZZFeatureMap:
        """Return a ZZFeatureMap configured for this block."""
        return ZZFeatureMap(feature_dimension=self.num_qubits, reps=self.feature_map_reps)

    @staticmethod
    def _ghost_module(num_qubits: int, reps: int) -> QuantumCircuit:
        """Lightweight, fixed-parameter "ghost" entangling ansatz.

        Pattern:
            - For each repetition:
                * Apply RY(pi/4) to all qubits
                * Apply a linear chain of CNOTs
        """
        qc = QuantumCircuit(num_qubits)
        for _ in range(reps):
            for q in range(num_qubits):
                qc.ry(np.pi / 4.0, q)
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Create the chosen ansatz circuit."""
        if self.mode == "real_amplitudes":
            return RealAmplitudes(num_qubits=self.num_qubits, reps=self.ansatz_reps)
        # default: "ghost"
        return self._ghost_module(self.num_qubits, self.ansatz_reps)

    # endregion

    # region ---- Public API ----
    def build(self) -> Tuple[ZZFeatureMap, QuantumCircuit, QuantumCircuit]:
        """
        Build and return (feature_map, ansatz, combined_circuit).

        The combined circuit is:
            QuantumCircuit(num_qubits) ∘ feature_map ∘ ansatz
        """
        feature_map = self._build_feature_map()
        ansatz = self._build_ansatz()

        qc = QuantumCircuit(self.num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        return feature_map, ansatz, qc
    # endregion


def create_quantum_ghost_block(
    num_qubits: int,
    feature_map_reps: int = DEFAULT_FEATURE_MAP_REPS,
    ansatz_reps: int = DEFAULT_ANSATZ_REPS,
    mode: Literal["ghost", "real_amplitudes"] = "ghost",
) -> tuple[ZZFeatureMap, QuantumCircuit, QuantumCircuit]:
    """
    Convenience function matching the original API idea.

    Returns
    -------
    (feature_map, ansatz, combined_circuit)
    """
    block = QuantumGhostBlock(
        num_qubits=num_qubits,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        mode=mode,
    )
    return block.build()
