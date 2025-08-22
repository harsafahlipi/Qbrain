
"""
Working QRAM implementation (simplified) for Qiskit-based pipelines.
"""

from __future__ import annotations

# region ========== Stdlib / Typing ==========
from typing import List, Sequence
# endregion

# region ========== Third-Party ==========
import numpy as np
from qiskit import QuantumCircuit
# endregion


class WorkingQRAM:
    """Simplified QRAM that encodes classical data conditioned on address qubits.

    Notes:
        - This is NOT a full QRAM data structure. It provides a pragmatic, circuit-level
          data loading approach suitable for toy experiments and ML pipelines.
        - Address lines control which data rotations are activated on data qubits.
    """

    def __init__(self, data_size: int, address_qubits: int, data_qubits: int) -> None:
        self.data_size = data_size
        self.address_qubits = address_qubits
        self.data_qubits = data_qubits
        self.total_qubits = address_qubits + data_qubits

        if 2 ** address_qubits < data_size:
            raise ValueError("Insufficient address qubits for data size.")

        # Classical data array stored independently of autograd
        self._data: np.ndarray | None = None

    # ---- Data Management ----
    def store_data(self, data: Sequence[float]) -> None:
        """Store classical data (detached; coerced to fixed length)."""
        arr = np.asarray(data, dtype=float).flatten()
        if len(arr) > self.data_size:
            arr = arr[: self.data_size]
        elif len(arr) < self.data_size:
            arr = np.pad(arr, (0, self.data_size - len(arr)), mode="constant")
        self._data = arr

    # ---- Circuit Generation ----
    def create_qram_circuit_with_input(self, input_params: Sequence) -> QuantumCircuit:
        """Build a circuit that uses input params to prepare the address superposition."""
        qc = QuantumCircuit(self.total_qubits)

        addr = list(range(self.address_qubits))
        data = list(range(self.address_qubits, self.total_qubits))

        # Parametric address preparation (Ry); default to Hadamard if not enough params
        for i, q in enumerate(addr):
            if i < len(input_params):
                qc.ry(input_params[i], q)
            else:
                qc.h(q)

        # Data loading via controlled rotations
        if self._data is not None:
            limit = min(self.data_size, len(self._data))
            for address in range(limit):
                self._add_controlled_encoding(qc, address, float(self._data[address]), addr, data)

        return qc

    def _add_controlled_encoding(
        self,
        qc: QuantumCircuit,
        address: int,
        data_val: float,
        address_reg: List[int],
        data_reg: List[int],
    ) -> None:
        """Apply simple controlled rotations for a given address pattern.

        Implementation details:
            - For 1 control -> CX on a single data qubit after an RY rotation.
            - For 2 controls -> CCX on the first data qubit after an RY rotation.
            - For >2 controls -> apply RY and fan-in with CX from the first two controls (approx).
            - Additional data qubits receive scaled phases with at most two controls.
        """
        n_addr = len(address_reg)
        # Address bit pattern (little-endian order)
        addr_bits = [(address >> i) & 1 for i in range(n_addr)]

        # X on address lines that should be |0> to match |address>
        flipped = []
        for i, bit in enumerate(addr_bits):
            if bit == 0:
                qc.x(address_reg[i])
                flipped.append(i)

        # Rotation angle (mod 2π)
        theta = float(abs(data_val) % (2.0 * np.pi))

        if n_addr == 1:
            qc.ry(theta, data_reg[0])
            qc.cx(address_reg[0], data_reg[0])
        elif n_addr == 2:
            qc.ry(theta, data_reg[0])
            qc.ccx(address_reg[0], address_reg[1], data_reg[0])
        else:
            qc.ry(theta, data_reg[0])
            for ctrl in address_reg[:2]:
                qc.cx(ctrl, data_reg[0])

        # Encode on additional data qubits (scaled)
        for i in range(1, min(len(data_reg), 2)):
            phase = theta * (i + 1) / max(1, len(data_reg))
            qc.ry(phase, data_reg[i])
            if n_addr >= 1:
                qc.cx(address_reg[0], data_reg[i])
            if n_addr >= 2:
                qc.cx(address_reg[1], data_reg[i])

        # Uncompute X flips
        for i in flipped:
            qc.x(address_reg[i])
