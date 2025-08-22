
"""
Variational Quantum Layer Normalization — Classical Variant
===========================================================

A simple, learnable normalization operating on a probability vector or the
squared amplitudes of a quantum state. This utility does not depend on
Qiskit execution and can be trained via manual gradient updates.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Tuple
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
# endregion


class VariationalQuantumLayerNorm:
    """Variational quantum layer normalization with learnable gamma/beta."""

    def __init__(self, n_qubits: int, learning_rate: float = 0.01) -> None:
        self.n_qubits = int(n_qubits)
        self.learning_rate = float(learning_rate)
        self.state_dim = 2 ** self.n_qubits

        # Trainable parameters (numpy; external optimizer can wrap if needed)
        rng = np.random.default_rng()
        self.gamma = rng.uniform(0.1, 2.0, self.state_dim).astype(np.float64)
        self.beta = rng.uniform(-0.5, 0.5, self.state_dim).astype(np.float64)

    @staticmethod
    def quantum_mean_variance(probabilities: np.ndarray) -> Tuple[float, float]:
        """Compute mean/variance over computational basis indices."""
        probs = np.asarray(probabilities, dtype=np.float64)
        probs = np.abs(probs) ** 2 if np.iscomplexobj(probs) else probs
        s = probs.sum()
        if s <= 0:
            raise ValueError("Probability vector sum must be positive.")
        probs = probs / s

        idx = np.arange(probs.size, dtype=np.float64)
        q_mean = float((probs * idx).sum())
        q_var = float((probs * (idx - q_mean) ** 2).sum())
        return q_mean, q_var

    def forward(self, input_state: np.ndarray) -> np.ndarray:
        """Normalize an input probability (or state amplitudes) vector."""
        probs = np.asarray(input_state, dtype=np.float64)
        probs = np.abs(probs) ** 2 if np.iscomplexobj(probs) else probs

        q_mean, q_var = self.quantum_mean_variance(probs)
        eps = 1e-8
        normalized = (probs - q_mean) / np.sqrt(q_var + eps)
        out = self.gamma * normalized + self.beta
        return out.astype(np.float64)

    def update_parameters(self, grad_gamma: np.ndarray, grad_beta: np.ndarray) -> None:
        """Apply a single gradient step on gamma/beta."""
        self.gamma = self.gamma - self.learning_rate * np.asarray(grad_gamma, dtype=np.float64)
        self.beta = self.beta - self.learning_rate * np.asarray(grad_beta, dtype=np.float64)
