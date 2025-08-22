
"""
Ansatz factory for QViT.
"""

from __future__ import annotations
from qiskit.circuit.library import RealAmplitudes


def create_ansatz(n_qubits: int, reps: int = 2) -> RealAmplitudes:
    """Create a RealAmplitudes ansatz with given repetitions."""
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    if reps <= 0:
        raise ValueError("reps must be positive.")
    return RealAmplitudes(num_qubits=n_qubits, reps=reps)
