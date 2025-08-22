
from __future__ import annotations
import numpy as np, torch
import torch.nn as nn
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    _HAS_Q = True
except Exception:
    _HAS_Q = False
    QuantumCircuit = None  # type: ignore
    transpile = None       # type: ignore
    AerSimulator = None    # type: ignore

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, shots: int = 1000, use_quantum_eval: bool = False) -> None:
        super().__init__()
        self.n_qubits, self.n_layers, self.shots, self.use_quantum_eval = n_qubits, n_layers, shots, use_quantum_eval
        total = n_qubits + (n_qubits * 2 * n_layers)
        self.weights = nn.Parameter(torch.randn(total) * 0.1)
        self._sim = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.weights[: self.n_qubits].unsqueeze(0).expand(x.size(0), -1)
        return torch.tanh(torch.tanh(x) * np.pi + 0.1 * bias)
