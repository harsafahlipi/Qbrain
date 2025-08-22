
"""
Quantum circuit builders (amplitude encoding, swap test).
"""

from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit

from .utils import require_vector1d


def amplitude_encode(vector: np.ndarray) -> QuantumCircuit:
    """Amplitude-encode a 1D vector into a fresh circuit (with zero-padding)."""
    require_vector1d("vector", vector)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Input vector cannot be all zeros.")
    vec = (vector / norm).astype(float, copy=False)
    n_qubits = int(np.ceil(np.log2(vec.size)))
    padded = np.pad(vec, (0, 2**n_qubits - vec.size))
    qc = QuantumCircuit(n_qubits)
    qc.initialize(padded, range(n_qubits), normalize=True)
    return qc


def swap_test(qc1: QuantumCircuit, qc2: QuantumCircuit) -> QuantumCircuit:
    """Create a swap-test circuit measuring the ancilla."""
    if qc1.num_qubits != qc2.num_qubits:
        raise ValueError("Circuits must have same number of qubits.")
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
