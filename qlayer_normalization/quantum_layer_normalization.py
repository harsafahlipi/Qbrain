
"""
Quantum Layer Normalization — Core
=================================

Implements a quantum-inspired layer normalization using parameterized quantum
circuits to compute normalization statistics and apply learnable scaling
(gamma) and shifting (beta).

Design:
- English docstrings, type hints, DRY/KISS/SOLID
- No prints/side effects; silent, deterministic fallbacks
- Region markers to separate concerns
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Dict, List, Optional
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

# Estimator may live in qiskit.primitives (modern) or be unavailable.
try:
    from qiskit.primitives import Estimator  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    Estimator = None  # type: ignore
# endregion

# region ========== Internal Utilities / Config ==========
from .utils import DEFAULT_LAYERS, require_positive, build_z_observables
# endregion


class QuantumLayerNormalization:
    """
    Quantum Layer Normalization using parameterized quantum circuits.

    Steps:
        1) Encode inputs via Ry rotations.
        2) Entangle to mix statistics.
        3) Apply learnable gamma/beta rotations.
        4) Measure Z-expectations as normalization outputs.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = DEFAULT_LAYERS,
        backend: Optional[AerSimulator] = None,
    ) -> None:
        """Initialize QLN with qubit count and repetition depth."""
        require_positive("n_qubits", n_qubits)
        require_positive("n_layers", n_layers)

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Learnable parameters (classical; caller manages values)
        self.gamma_params = ParameterVector("gamma", n_qubits * n_layers)  # Scale
        self.beta_params = ParameterVector("beta", n_qubits * n_layers)    # Shift

        # Simulation backend (for fallback)
        self.backend = backend or AerSimulator(method="statevector")

        # Optional estimator primitive (if available in the environment)
        self._estimator = Estimator() if Estimator is not None else None  # type: ignore

    # ---- Circuit builder ----
    def create_normalization_circuit(self, input_params: ParameterVector) -> QuantumCircuit:
        """Create the layer-normalization circuit for given input parameters."""
        if len(input_params) != self.n_qubits:
            raise ValueError(f"input_params length must be {self.n_qubits}.")

        qr = QuantumRegister(self.n_qubits, "q")
        qc = QuantumCircuit(qr)

        # Input encoding
        for i in range(self.n_qubits):
            qc.ry(input_params[i], qr[i])

        # Normalization blocks
        for layer in range(self.n_layers):
            # Entanglement
            for i in range(self.n_qubits - 1):
                qc.cx(qr[i], qr[i + 1])

            # Learnable scaling (gamma) and shifting (beta)
            for i in range(self.n_qubits):
                idx = layer * self.n_qubits + i
                qc.rz(self.gamma_params[idx], qr[i])
                qc.ry(self.beta_params[idx], qr[i])

        return qc

    # ---- Expectation utilities ----
    def _expvals_estimator(
        self,
        circuit: QuantumCircuit,
        observables: List[SparsePauliOp],
        param_values: Dict[object, float],
    ) -> Optional[List[float]]:
        """Try to compute expectation values via Estimator primitive.

        Returns None if Estimator is unavailable or execution fails.
        """
        if self._estimator is None:
            return None
        try:
            bound = circuit.assign_parameters(param_values, inplace=False)
            # Build "pubs" as (circuit, observable) tuples (EstimatorV2 style)
            pubs = [(bound, obs) for obs in observables]
            job = self._estimator.run(pubs)  # type: ignore[arg-type]
            result = job.result()
            # Depending on qiskit version, result may be list-like or object with .values
            expvals: List[float] = []
            for item in result:  # type: ignore[assignment]
                # Try common access patterns
                v = getattr(item, "data", None)
                if v is not None and hasattr(v, "evs"):
                    expvals.append(float(v.evs))
                elif hasattr(item, "value"):
                    expvals.append(float(item.value))  # type: ignore[arg-type]
                else:
                    # Fallback: attempt to cast
                    expvals.append(float(item))  # type: ignore[arg-type]
            return expvals
        except Exception:
            return None

    def _expvals_statevector(
        self,
        circuit: QuantumCircuit,
        observables: List[SparsePauliOp],
        param_values: Dict[object, float],
    ) -> List[float]:
        """Compute expectation values via analytic statevector simulation (fallback)."""
        bound = circuit.assign_parameters(param_values, inplace=False)
        # Prefer analytic construction when available
        try:
            state = Statevector.from_instruction(bound)
        except Exception:
            tqc = transpile(bound, self.backend, optimization_level=2)
            res = self.backend.run(tqc, shots=1).result()
            state = res.get_statevector()
        expvals: List[float] = []
        for obs in observables:
            ev = Statevector(state).expectation_value(obs)  # type: ignore[arg-type]
            expvals.append(float(np.real(ev)))
        return expvals

    def compute_expectation_values(
        self,
        circuit: QuantumCircuit,
        observables: List[SparsePauliOp],
        param_values: Dict[object, float],
    ) -> List[float]:
        """Compute expectation values using Estimator if possible, else statevector fallback."""
        vals = self._expvals_estimator(circuit, observables, param_values)
        if vals is not None:
            return vals
        return self._expvals_statevector(circuit, observables, param_values)

    # ---- Public normalization API ----
    def normalize_quantum_state(
        self,
        input_data: np.ndarray,
        gamma_values: np.ndarray,
        beta_values: np.ndarray,
    ) -> np.ndarray:
        """Apply quantum layer normalization to input angles.

        Args:
            input_data: Input angles for Ry encoders (shape: [n_qubits]).
            gamma_values: Scale parameters (shape: [n_qubits * n_layers]).
            beta_values: Shift parameters (shape: [n_qubits * n_layers]).

        Returns:
            np.ndarray of Z-expectations per qubit.
        """
        input_data = np.asarray(input_data, dtype=np.float64).ravel()
        gamma_values = np.asarray(gamma_values, dtype=np.float64).ravel()
        beta_values = np.asarray(beta_values, dtype=np.float64).ravel()

        if input_data.size != self.n_qubits:
            raise ValueError(f"input_data must have length {self.n_qubits}.")
        if gamma_values.size != self.n_qubits * self.n_layers:
            raise ValueError("gamma_values length mismatch.")
        if beta_values.size != self.n_qubits * self.n_layers:
            raise ValueError("beta_values length mismatch.")

        input_params = ParameterVector("input", self.n_qubits)
        circuit = self.create_normalization_circuit(input_params)
        observables = build_z_observables(self.n_qubits)

        # Parameter bindings
        bindings: Dict[object, float] = {}
        for i, p in enumerate(input_params):
            bindings[p] = float(input_data[i])
        for i, p in enumerate(self.gamma_params):
            bindings[p] = float(gamma_values[i])
        for i, p in enumerate(self.beta_params):
            bindings[p] = float(beta_values[i])

        expvals = self.compute_expectation_values(circuit, observables, bindings)
        return np.asarray(expvals, dtype=np.float32)
