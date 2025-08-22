
"""
Expectation utilities using Statevector.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import List
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit import QuantumCircuit
# endregion


def z_expectations(circuit: QuantumCircuit) -> np.ndarray:
    """
    Compute ⟨Z_i⟩ for each qubit i via statevector expectation values.
    """
    n = circuit.num_qubits
    # Simulate statevector directly from the circuit (supports initialize)
    psi = Statevector.from_instruction(circuit)

    exps: List[float] = []
    for i in range(n):
        pauli = ['I'] * n
        pauli[i] = 'Z'
        obs = SparsePauliOp.from_list([(''.join(pauli), 1.0)])
        val = psi.expectation_value(obs)
        exps.append(float(np.real(val)))
    return np.asarray(exps, dtype=float)
