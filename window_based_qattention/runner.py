
"""
Execution helpers (simulation).
"""

from __future__ import annotations
from typing import Dict
from qiskit import transpile
from qiskit_aer import AerSimulator

from .utils import DEFAULT_SHOTS


def measure_counts(qc, shots: int = DEFAULT_SHOTS) -> Dict[str, float]:
    """
    Transpile and simulate the circuit using AerSimulator.
    Returns normalized counts as probabilities.
    """
    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend, optimization_level=3)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    total = float(shots) if shots else 1.0
    return {state: cnt / total for state, cnt in counts.items()}
