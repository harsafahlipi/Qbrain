
"""
Quantum Transformer Decoder Block — circuit builder.
"""

from __future__ import annotations
from typing import Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from .utils import require_positive


def create_quantum_decoder_block(n_qubits: int, n_layers: int = 4) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build a decoder-like quantum circuit with angle encoding + alternating entanglement/param layers.

    Args:
        n_qubits: number of qubits (and input features).
        n_layers: number of attention/FFN-like repetitions.

    Returns:
        (circuit, input_params, weight_params)
    """
    require_positive("n_qubits", n_qubits)
    require_positive("n_layers", n_layers)

    input_params = ParameterVector('x', n_qubits)                 # data encoding
    weight_params = ParameterVector('w', n_qubits * 2 * n_layers)  # per-layer RY/RZ per qubit
    circ = QuantumCircuit(n_qubits)

    # Angle encoding of inputs
    for i in range(n_qubits):
        circ.ry(input_params[i], i)

    # Alternating attention-like entanglement and FFN-like rotations
    p = 0
    for _ in range(n_layers):
        # Linear + ring entanglement
        for i in range(n_qubits - 1):
            circ.cx(i, i + 1)
        if n_qubits > 1:
            circ.cx(n_qubits - 1, 0)
        circ.barrier()

        # Feed-forward-like parameterized layer
        for i in range(n_qubits):
            circ.ry(weight_params[p], i); p += 1
            circ.rz(weight_params[p], i); p += 1
        circ.barrier()

    return circ, input_params, weight_params
