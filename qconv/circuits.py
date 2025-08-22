
"""
Circuit builders for qConv.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Sequence
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
from qiskit import QuantumCircuit
# endregion

# region ========== Internal ==========
from .utils import require_vector1d, l2_normalize, next_pow2_qubits, pad_to_length
# endregion


def amplitude_encoding_circuit(patch: np.ndarray) -> QuantumCircuit:
    """
    Build an amplitude-encoding circuit for a 1D patch (normalized + padded).
    Ensures at least 1 qubit.
    """
    require_vector1d("patch", patch)
    patch = l2_normalize(patch)
    n_qubits = next_pow2_qubits(patch.size)
    target_len = 2 ** n_qubits
    amps = pad_to_length(patch, target_len)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(amps, list(range(n_qubits)))  # already normalized
    return qc


def apply_filter_and_entanglement(qc: QuantumCircuit, weights: Sequence[float]) -> QuantumCircuit:
    """
    Apply parameterized RY rotations per qubit (weights repeating if needed)
    and a linear entanglement (CNOT chain).
    Returns the mutated circuit for convenience.
    """
    n = qc.num_qubits
    # RY layer
    for i in range(n):
        theta = float(weights[i % len(weights)]) if len(weights) > 0 else 0.0
        qc.ry(theta, i)
    # CNOT chain
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc
