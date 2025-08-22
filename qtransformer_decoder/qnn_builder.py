
"""
QNN builder for the Quantum Transformer Decoder circuit.
"""

from __future__ import annotations
from typing import Any

from .decoder_block import create_quantum_decoder_block
from .utils import build_z_observables, get_estimator

# Qiskit ML (optional)
try:
    from qiskit_machine_learning.neural_networks import EstimatorQNN  # type: ignore
except Exception as _e:  # pragma: no cover
    EstimatorQNN = None  # type: ignore


def create_qnn(n_qubits: int, n_layers: int = 4) -> Any:
    """Create an EstimatorQNN for the decoder circuit (if Qiskit ML primitives are available)."""
    circ, x_params, w_params = create_quantum_decoder_block(n_qubits, n_layers)
    observables = build_z_observables(n_qubits)
    est = get_estimator()
    if est is None or EstimatorQNN is None:
        raise RuntimeError("Estimator or EstimatorQNN not available in this environment.")
    qnn = EstimatorQNN(
        circuit=circ,
        input_params=list(x_params),
        weight_params=list(w_params),
        observables=observables,
        estimator=est,
    )
    return qnn
