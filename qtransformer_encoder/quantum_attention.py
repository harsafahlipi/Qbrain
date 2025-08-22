
"""
QuantumAttention — Qiskit-based attention primitive with safe fallbacks.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Optional, Tuple
# endregion

# region ========== Third-Party Imports ==========
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
# Optional ML primitives
try:
    from qiskit_machine_learning.neural_networks import EstimatorQNN  # type: ignore
    from qiskit_machine_learning.connectors import TorchConnector    # type: ignore
except Exception:  # pragma: no cover
    EstimatorQNN = None  # type: ignore
    TorchConnector = None  # type: ignore
# endregion

# region ========== Internal Utilities ==========
from .utils import build_z_observables, build_z0_observable, get_estimator
# endregion


class QuantumAttention(nn.Module):
    """
    Quantum attention module.

    - Builds two circuits:
        * attention_circuit: outputs a single scalar attention score per token
          via Z on qubit 0.
        * value_circuit: outputs a vector of length `num_qubits` via Z on each qubit.
    - If Qiskit ML primitives are unavailable, falls back to small classical nets.
    """

    def __init__(self, num_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.num_qubits = int(num_qubits)
        self.depth = int(depth)

        # If either EstimatorQNN or TorchConnector is unavailable, use classical fallback
        self._use_quantum = (EstimatorQNN is not None) and (TorchConnector is not None)

        if self._use_quantum:
            try:
                # Shared templates
                self.feature_map = ZZFeatureMap(self.num_qubits, reps=1)
                self.var_form = RealAmplitudes(self.num_qubits, reps=self.depth)

                # Attention circuit (scalar output)
                attn_circ = QuantumCircuit(self.num_qubits)
                attn_circ.compose(self.feature_map, inplace=True)
                attn_circ.compose(self.var_form, inplace=True)

                # Value circuit (vector output)
                val_circ = QuantumCircuit(self.num_qubits)
                val_circ.compose(self.feature_map, inplace=True)
                val_circ.compose(self.var_form, inplace=True)

                # Observables
                obs_attn = [build_z0_observable(self.num_qubits)]           # -> scalar
                obs_vals = build_z_observables(self.num_qubits)            # -> vector

                est = get_estimator()

                # QNNs
                self._attn_qnn = EstimatorQNN(
                    circuit=attn_circ,
                    input_params=list(self.feature_map.parameters),
                    weight_params=list(self.var_form.parameters),
                    observables=obs_attn,
                    estimator=est,
                )
                self._val_qnn = EstimatorQNN(
                    circuit=val_circ,
                    input_params=list(self.feature_map.parameters),
                    weight_params=list(self.var_form.parameters),
                    observables=obs_vals,
                    estimator=est,
                )

                # Connectors
                self.attention_layer = TorchConnector(self._attn_qnn)
                self.value_layer = TorchConnector(self._val_qnn)

            except Exception:
                # Quantum path failed; fall back
                self._use_quantum = False

        if not self._use_quantum:
            # Classical surrogates
            self.attention_layer = nn.Sequential(
                nn.Linear(self.num_qubits, 1),
                nn.Tanh(),
            )
            self.value_layer = nn.Sequential(
                nn.Linear(self.num_qubits, self.num_qubits),
                nn.Tanh(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim) where feature_dim == num_qubits

        Returns:
            (batch, seq_len, num_qubits) — attention-weighted values.
        """
        b, s, f = x.shape
        x_flat = x.reshape(-1, f)  # (b*s, f)

        attn_scores = self.attention_layer(x_flat)     # (b*s, 1)
        values = self.value_layer(x_flat)              # (b*s, num_qubits)

        attn_scores = attn_scores.view(b, s, 1)
        values = values.view(b, s, self.num_qubits)

        return attn_scores * values
