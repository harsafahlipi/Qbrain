
"""
Quantum circuit builders: amplitude encoding and swap test.
"""

from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit

from .utils import require_nonempty_vector


def amplitude_encode(vector: np.ndarray) -> QuantumCircuit:
    """Amplitude-encode a 1D vector into a fresh circuit (with zero-padding)."""
    require_nonempty_vector("vector", vector)
    n_qubits = int(np.ceil(np.log2(vector.size)))
    padded_size = 2 ** n_qubits
    vec = np.pad(vector.astype(np.float64, copy=False), (0, padded_size - vector.size))
    vec = vec / np.linalg.norm(vec)
    qc = QuantumCircuit(n_qubits)
    qc.initialize(vec, list(range(n_qubits)))
    return qc


def swap_test(qc1: QuantumCircuit, qc2: QuantumCircuit) -> QuantumCircuit:
    """Build a swap-test circuit comparing the states prepared by qc1 and qc2."""
    if qc1.num_qubits != qc2.num_qubits:
        raise ValueError("Circuits must have the same number of qubits.")
    n = qc1.num_qubits
    qc = QuantumCircuit(2 * n + 1, 1)
    qc.compose(qc1, qubits=range(1, n + 1), inplace=True)
    qc.compose(qc2, qubits=range(n + 1, 2 * n + 1), inplace=True)
    qc.h(0)
    for i in range(n):
        qc.cswap(0, i + 1, n + i + 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc
