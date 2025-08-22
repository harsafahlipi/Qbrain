
"""
FastQuantumLayer — a lightweight quantum layer with classical fallback.
"""

from __future__ import annotations

# region ========== Imports ==========
from typing import Optional
import numpy as np
import torch
from torch import nn

from .utils import QUANTUM_AVAILABLE
# Guarded imports for quantum stack
try:
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    _QML_READY = True
except Exception:  # pragma: no cover - optional dependency
    _QML_READY = False
# endregion


class FastQuantumLayer(nn.Module):
    """
    Lightweight quantum layer optimized for speed with a classical fallback.

    If the quantum stack is unavailable or disabled, the layer automatically
    falls back to a small classical network that mimics I/O shape.
    """

    def __init__(self, n_qubits: int = 2, use_quantum: bool = True) -> None:
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.use_quantum = bool(use_quantum and QUANTUM_AVAILABLE and _QML_READY)

        if self.use_quantum:
            try:
                self.quantum_layer = self._create_fast_quantum_layer()
            except Exception:
                # Fallback to classical path if quantum init fails
                self.use_quantum = False
                self.quantum_layer = self._create_classical_fallback()
        else:
            self.quantum_layer = self._create_classical_fallback()

    # ------------------- Builders -------------------
    @staticmethod
    def _build_quantum_circuit() -> "QuantumCircuit":
        """Create a minimal 2-qubit circuit with parameterized RY and entanglement."""
        qc = QuantumCircuit(2)
        theta = Parameter("theta")
        phi = Parameter("phi")
        qc.ry(theta, 0)
        qc.ry(phi, 1)
        qc.cx(0, 1)
        qc.ry(theta * 0.5, 0)
        qc.ry(phi * 0.5, 1)
        return qc

    def _create_fast_quantum_layer(self) -> nn.Module:
        """Create an EstimatorQNN wrapped with TorchConnector."""
        if not (_QML_READY and QUANTUM_AVAILABLE):
            raise RuntimeError("Quantum stack not available.")
        qc = self._build_quantum_circuit()
        obs1 = SparsePauliOp.from_list([("ZI", 1.0)])
        obs2 = SparsePauliOp.from_list([("IZ", 1.0)])
        estimator = Estimator()  # default estimator; fast on simulators
        qnn = EstimatorQNN(
            circuit=qc,
            observables=[obs1, obs2],
            input_params=list(qc.parameters),
            weight_params=[],
            estimator=estimator,
        )
        return TorchConnector(qnn)

    def _create_classical_fallback(self) -> nn.Module:
        """Small MLP that maps 2 inputs to n_qubits outputs."""
        width = max(2, self.n_qubits)
        return nn.Sequential(
            nn.Linear(2, width),
            nn.Tanh(),
            nn.Linear(width, self.n_qubits),
            nn.Tanh(),
        )

    # ------------------- Forward -------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Uses quantum path if enabled; otherwise uses the classical fallback.

        Notes
        -----
        - Only the first 2 features are consumed by this layer.
        - Output is zero-padded if quantum connector returns fewer than `n_qubits` features.
        """
        x2 = x[:, :2]
        if self.use_quantum:
            try:
                xq = torch.tanh(x2) * (np.pi / 2.0)
                result = self.quantum_layer(xq)
                # Ensure output width
                if result.size(1) < self.n_qubits:
                    pad = torch.zeros(x.size(0), self.n_qubits - result.size(1), device=x.device, dtype=result.dtype)
                    result = torch.cat([result, pad], dim=1)
                return result
            except Exception:
                # Switch permanently to classical if quantum forward fails
                self.use_quantum = False
                self.quantum_layer = self._create_classical_fallback().to(x.device)

        return self.quantum_layer(x2)
