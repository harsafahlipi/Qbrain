
"""
QPE (Quantum Positional Encoding) — Core Module
===============================================

Provides:
- QuantumSinusoidalPositionalEncoding: classical sinusoidal PE embedded into a quantum state.
- QuantumLearnablePositionalEncoding: learnable quantum PE with parameterized circuits (PyTorch).

Design:
- Docstrings, type hints, KISS/DRY/SOLID
- Region markers for readability
- Backend injection for simulators, no prints or side effects
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import List, Sequence, Optional
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

import torch
import torch.nn as nn
# endregion

# region ========== Internal Utilities / Config ==========
from .utils import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_D_MODEL,
    DEFAULT_BASE,
    DEFAULT_SIM_METHOD,
    DEFAULT_MAX_THREADS,
    require_positive,
    require_power_of_two,
    l2_normalize,
)
# endregion


# region ========== QuantumSinusoidalPositionalEncoding ==========
class QuantumSinusoidalPositionalEncoding:
    """Quantum implementation of sinusoidal positional encoding.

    Maps classical positional information into a quantum state via amplitude initialization.
    """

    def __init__(
        self,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        d_model: int = DEFAULT_D_MODEL,
        base: int = DEFAULT_BASE,
        backend: Optional[AerSimulator] = None,
        sim_method: str = DEFAULT_SIM_METHOD,
        max_threads: int = DEFAULT_MAX_THREADS,
    ) -> None:
        """Initialize the encoding module.

        Args:
            max_seq_len: Maximum sequence length.
            d_model: Embedding dimension (must be a power of two for amplitude encoding).
            base: Base used in sinusoidal frequency scaling.
            backend: Optional preconfigured AerSimulator; if None, one is created.
            sim_method: Simulation method to use for AerSimulator.
            max_threads: Max parallel threads for AerSimulator (if created here).
        """
        require_positive("max_seq_len", max_seq_len)
        require_power_of_two("d_model", d_model)

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.base = base
        self.n_qubits = int(np.log2(d_model))
        self.backend = backend or AerSimulator(method=sim_method, max_parallel_threads=max_threads)

        # Precompute classical encodings (shape: [max_seq_len, d_model])
        self._pos_encodings = self._compute_classical_encodings()

    # ---- Classical sinusoidal PE ----
    def _compute_classical_encodings(self) -> np.ndarray:
        """Compute classical sinusoidal positional encodings (shape: [L, D])."""
        L, D = self.max_seq_len, self.d_model
        pos = np.arange(L, dtype=np.float64)[:, None]       # (L, 1)
        i = np.arange(D, dtype=np.float64)[None, :]         # (1, D)
        div_term = np.exp(-(np.log(self.base) / D) * (i - (i % 2)))  # Only even indices affect frequency
        enc = np.zeros((L, D), dtype=np.float64)
        enc[:, 0::2] = np.sin(pos * div_term[:, 0::2]) if D > 0 else enc[:, 0::2]
        enc[:, 1::2] = np.cos(pos * div_term[:, 1::2]) if D > 1 else enc[:, 1::2]
        return enc.astype(np.float32)

    # ---- Circuit builder ----
    def create_quantum_circuit(self, position: int) -> QuantumCircuit:
        """Create a quantum circuit that encodes the positional vector at `position`.

        Uses amplitude initialization; if initialization is not feasible, falls back
        to simple per-qubit Ry rotations derived from the (clipped) amplitudes.
        """
        if not (0 <= position < self.max_seq_len):
            raise ValueError(f"position must be in [0, {self.max_seq_len-1}], got {position}.")

        classical = self._pos_encodings[position]           # (D,)
        amplitudes = l2_normalize(classical)                # (D,)

        qc = QuantumCircuit(self.n_qubits)
        try:
            qc.initialize(amplitudes, range(self.n_qubits))
        except Exception:
            # Fallback: approximate with local Ry rotations using first n_qubits components
            subset = amplitudes[: self.n_qubits]
            for i, amp in enumerate(subset):
                theta = 2.0 * np.arcsin(np.clip(abs(amp), 0.0, 1.0))
                qc.ry(theta, i)
        return qc

    # ---- Batch encoder ----
    def encode_sequence(self, sequence_positions: Sequence[int]) -> List[QuantumCircuit]:
        """Return a list of circuits, one per input position."""
        circuits: List[QuantumCircuit] = []
        for pos in sequence_positions:
            circuits.append(self.create_quantum_circuit(int(pos)))
        return circuits
# endregion


# region ========== QuantumLearnablePositionalEncoding ==========
class QuantumLearnablePositionalEncoding(nn.Module):
    """Learnable quantum positional encoding using parameterized quantum circuits."""

    def __init__(
        self,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        n_qubits: int = 3,
        n_layers: int = 2,
        backend: Optional[AerSimulator] = None,
        sim_method: str = DEFAULT_SIM_METHOD,
        max_threads: int = DEFAULT_MAX_THREADS,
    ) -> None:
        """Initialize the learnable quantum PE.

        Args:
            max_seq_len: Maximum sequence length (number of distinct positions).
            n_qubits: Number of qubits in the parameterized circuit.
            n_layers: Number of repeated RX/RY/RZ layers + entanglers.
            backend: Optional AerSimulator; if None, one is created.
            sim_method: Simulation method for AerSimulator (if created).
            max_threads: Max parallel threads for AerSimulator.
        """
        super().__init__()
        require_positive("max_seq_len", max_seq_len)
        require_positive("n_qubits", n_qubits)
        require_positive("n_layers", n_layers)

        self.max_seq_len = max_seq_len
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # One learnable parameter vector per position per layer: [n_qubits * 3]
        self._n_params_per_layer = n_qubits * 3
        self._theta = nn.Parameter(torch.randn(max_seq_len, n_layers, self._n_params_per_layer) * 0.1)

        self.backend = backend or AerSimulator(method=sim_method, max_parallel_threads=max_threads)

    # ---- Parameterized circuit ----
    def _create_parameterized_circuit(self, position: int) -> QuantumCircuit:
        """Create a parameterized circuit for a given position (using frozen params)."""
        if not (0 <= position < self.max_seq_len):
            raise ValueError(f"position must be in [0, {self.max_seq_len-1}], got {position}.")

        qc = QuantumCircuit(self.n_qubits)
        pos_params = self._theta[position].detach().cpu().numpy()  # (n_layers, n_qubits*3)

        for layer in range(self.n_layers):
            layer_params = pos_params[layer]
            idx = 0
            for q in range(self.n_qubits):
                qc.rx(layer_params[idx + 0], q)
                qc.ry(layer_params[idx + 1], q)
                qc.rz(layer_params[idx + 2], q)
                idx += 3
            # Ring entanglement (cx q->q+1)
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        return qc

    # ---- Statevector extraction ----
    def _get_state(self, circuit: QuantumCircuit) -> np.ndarray:
        """Return the statevector for `circuit` (prefers analytic construction)."""
        try:
            state = Statevector.from_instruction(circuit).data
            return np.asarray(state)
        except Exception:
            tqc = transpile(circuit, self.backend, optimization_level=2)
            result = self.backend.run(tqc, shots=1).result()
            statevector = result.get_statevector()
            return np.asarray(statevector)

    # ---- Forward ----
    def forward(self, positions: Sequence[int]) -> torch.Tensor:
        """Return a real-valued tensor of stacked statevectors for the given positions.

        The real part of each statevector is returned as a float32 tensor of shape:
            (len(positions), 2**n_qubits)
        """
        states = []
        for pos in positions:
            circ = self._create_parameterized_circuit(int(pos))
            sv = self._get_state(circ)
            states.append(torch.tensor(sv.real, dtype=torch.float32))
        return torch.stack(states, dim=0)
# endregion
