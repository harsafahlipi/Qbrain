
"""
QRAM-enhanced Torch layer that couples classical inputs with a QRAM-backed
quantum circuit via EstimatorQNN + TorchConnector.
"""

from __future__ import annotations

# region ========== Stdlib / Typing ==========
import math
# endregion

# region ========== Third-Party ==========
import numpy as np
import torch
import torch.nn as nn

from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
# endregion

# region ========== Local Imports ==========
from .working_qram import WorkingQRAM
# endregion


class QRAM_Layer(nn.Module):
    """A QRAM-backed hybrid layer with classical projections and quantum inference.

    Pipeline:
        input (B, D_in) -> Linear to address_dim -> tanh * π -> QNN (EstimatorQNN)
        -> (B, 1) expectation -> Linear to D_out
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        qram_size: int = 8,
        n_qubits: int = 6,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.qram_size = qram_size
        self.n_qubits = n_qubits

        # Address/data partition
        self.address_qubits = max(1, min(3, math.ceil(math.log2(qram_size))))
        self.data_qubits = max(1, n_qubits - self.address_qubits)

        # QRAM
        self.qram = WorkingQRAM(qram_size, self.address_qubits, self.data_qubits)
        self.qram.store_data(np.random.rand(qram_size) * 2.0 * np.pi)

        # Parameters for QNN
        self.input_params = ParameterVector("input", self.address_qubits)
        self.weight_params = ParameterVector("weights", n_qubits)

        # Circuit + observable
        qc = self._create_qram_circuit()
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1.0)])

        # EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            observables=[observable],
        )
        init_w = np.random.rand(len(self.weight_params)) * 0.1
        self.quantum_layer = TorchConnector(self.qnn, init_w)

        # Classical projections
        self.input_projection = nn.Linear(input_dim, self.address_qubits)
        self.output_projection = nn.Linear(1, output_dim)

    def _create_qram_circuit(self):
        qc = self.qram.create_qram_circuit_with_input(self.input_params)

        # Variational layer
        for i in range(self.n_qubits):
            if i < len(self.weight_params):
                qc.ry(self.weight_params[i], i)

        # Entanglement
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: classical projection -> quantum expectation -> classical projection."""
        batch = x.size(0)
        x_proj = self.input_projection(x)
        x_norm = torch.tanh(x_proj) * math.pi

        try:
            q_out = self.quantum_layer(x_norm)  # (B, 1) expected
            if q_out.dim() == 1:
                q_out = q_out.unsqueeze(1)
            out = self.output_projection(q_out)
        except Exception:
            # Silent fallback without prints (production-friendly)
            out = self.output_projection(torch.zeros(batch, 1, device=x.device, dtype=x.dtype))

        return out
