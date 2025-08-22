
"""
Circuit builders for QSAL/QSANN.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Iterable
# endregion

# region ========== Third-Party Imports ==========
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
# endregion


# region ========== Circuit recipes ==========
def build_feature_map(n_qubits: int, enc_depth: int, params: ParameterVector) -> QuantumCircuit:
    """
    Feature map:
      - Initial per-qubit RX, RY
      - For each of enc_depth layers:
          * ring of CX entanglers
          * per-qubit RY
    Param vector length must be n_qubits * (enc_depth + 2).
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0
    # Initial rotations
    for j in range(n_qubits):
        qc.rx(params[idx], j)
        qc.ry(params[idx + 1], j)
        idx += 2
    # Encoding layers
    for _ in range(enc_depth):
        for j in range(n_qubits):
            qc.cx(j, (j + 1) % n_qubits)
        for j in range(n_qubits):
            qc.ry(params[idx], j)
            idx += 1
    return qc


def build_ansatz(n_qubits: int, var_depth: int, params: ParameterVector) -> QuantumCircuit:
    """
    Ansatz:
      - Initial per-qubit RX, RY
      - For each of var_depth layers:
          * ring of CX entanglers
          * per-qubit RY
    Param vector length must be n_qubits * (var_depth + 2).
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0
    # Initial rotations
    for j in range(n_qubits):
        qc.rx(params[idx], j)
        qc.ry(params[idx + 1], j)
        idx += 2
    # Variational layers
    for _ in range(var_depth):
        for j in range(n_qubits):
            qc.cx(j, (j + 1) % n_qubits)
        for j in range(n_qubits):
            qc.ry(params[idx], j)
            idx += 1
    return qc


def build_qsal_circuit(kind: str, n_qubits: int, enc_depth: int, var_depth: int,
                       input_params: ParameterVector, weight_params: ParameterVector) -> QuantumCircuit:
    """
    Compose feature map + ansatz for a given kind in {"Q", "K", "V"}.
    """
    qc = QuantumCircuit(n_qubits)
    fm = build_feature_map(n_qubits, enc_depth, input_params)
    an = build_ansatz(n_qubits, var_depth, weight_params)
    qc.compose(fm, inplace=True)
    qc.compose(an, inplace=True)
    return qc
# endregion


__all__ = ["build_feature_map", "build_ansatz", "build_qsal_circuit"]
