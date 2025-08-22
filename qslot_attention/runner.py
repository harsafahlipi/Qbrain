
"""
Execution utilities for quantum circuits.
"""

from __future__ import annotations
from typing import Dict

from qiskit import transpile
from qiskit_aer import AerSimulator

from .utils import DEFAULT_SHOTS


def measure_counts(qc, shots: int = DEFAULT_SHOTS) -> Dict[str, float]:
    """
    Transpile and simulate the circuit using AerSimulator (statevector method).
    Returns normalized counts as probabilities over measured bitstrings.
    """
    backend = AerSimulator(method="statevector", max_parallel_threads=8)
    tqc = transpile(qc, backend, optimization_level=3)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    total = float(shots) if shots else 1.0
    return {state: cnt / total for state, cnt in counts.items()}
