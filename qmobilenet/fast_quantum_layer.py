
"""
FastQuantumLayer — Hybrid quantum/classical feature projector.
--------------------------------------------------------------
A lightweight 2-qubit EstimatorQNN (if available) wrapped for PyTorch.
Falls back to a small classical MLP when quantum execution is unavailable.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Optional
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

# Optional primitives and ML bindings
try:
    from qiskit.primitives import Estimator  # type: ignore
except Exception:  # pragma: no cover
    Estimator = None  # type: ignore

try:
    from qiskit_machine_learning.neural_networks import EstimatorQNN  # type: ignore
    from qiskit_machine_learning.connectors import TorchConnector    # type: ignore
except Exception:  # pragma: no cover
    EstimatorQNN = None  # type: ignore
    TorchConnector = None  # type: ignore
# endregion


class FastQuantumLayer(nn.Module):
    """
    Lightweight quantum layer optimized for speed.
    Integrates a small quantum circuit; silently falls back to classical MLP.
    """

    def __init__(self, n_qubits: int = 2, use_quantum: bool = True) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.use_quantum = bool(use_quantum) and (Estimator is not None) and (EstimatorQNN is not None) and (TorchConnector is not None)

        if self.use_quantum:
            try:
                self.quantum_layer = self._create_fast_quantum_layer()
            except Exception:
                self.use_quantum = False
                self.quantum_layer = self._create_classical_fallback()
        else:
            self.quantum_layer = self._create_classical_fallback()

    def _create_fast_quantum_layer(self):
        """Create a 2-qubit EstimatorQNN connected to PyTorch."""
        qc = QuantumCircuit(2)

        # Parameters
        theta = Parameter('theta')
        phi = Parameter('phi')

        # Minimal entangling circuit
        qc.ry(theta, 0)
        qc.ry(phi, 1)
        qc.cx(0, 1)
        qc.ry(theta * 0.5, 0)
        qc.ry(phi * 0.5, 1)

        # Observables
        obs1 = SparsePauliOp.from_list([('ZI', 1.0)])
        obs2 = SparsePauliOp.from_list([('IZ', 1.0)])
        observables = [obs1, obs2]

        est = Estimator()  # rely on default settings
        qnn = EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=[theta, phi],
            weight_params=[],
            estimator=est,
        )

        return TorchConnector(qnn)

    @staticmethod
    def _create_classical_fallback() -> nn.Module:
        """Classical MLP that mimics a small quantum projection."""
        return nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        - If quantum enabled: process first 2 features via QNN (inputs ∈ [-π/2, π/2]).
        - Else: run classical MLP on first 2 features.
        Output is padded/truncated to match `n_qubits`.
        """
        # Select first 2 features
        x2 = x[:, :2]
        # Normalize to suitable range
        x2 = torch.tanh(x2) * (np.pi / 2.0)

        try:
            out = self.quantum_layer(x2)  # type: ignore[operator]
        except Exception:
            # Silent fallback for runtime issues
            self.use_quantum = False
            self.quantum_layer = self._create_classical_fallback().to(x.device)
            out = self.quantum_layer(x2)

        # Ensure output dimension == n_qubits via padding/truncation
        if out.size(1) < self.n_qubits:
            pad = torch.zeros(x.size(0), self.n_qubits - out.size(1), device=x.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=1)
        elif out.size(1) > self.n_qubits:
            out = out[:, : self.n_qubits]

        return out
