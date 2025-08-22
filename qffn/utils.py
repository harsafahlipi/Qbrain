
"""
Utilities and configuration for the QFFN (Quantum Feed-Forward Network) module.
"""

from __future__ import annotations
from typing import Optional
import os

# ---- Module-level configuration ("globals") ----
DEFAULT_INPUT_DIM: int = 784
DEFAULT_HIDDEN_DIM: int = 32
DEFAULT_OUTPUT_DIM: int = 10
DEFAULT_QUANTUM_RATIO: float = 0.25

# QUANTUM_AVAILABLE can be toggled via environment variable or runtime probing.
# If set to "0" explicitly, disables quantum even if dependencies are installed.
ENV_DISABLE_QUANTUM = os.getenv("DISABLE_QUANTUM", "0") == "1"


def quantum_available() -> bool:
    """Return True if quantum stack appears available and not explicitly disabled."""
    if ENV_DISABLE_QUANTUM:
        return False
    try:
        # Light-weight import probes
        import qiskit  # noqa: F401
        from qiskit.primitives import Estimator  # noqa: F401
        from qiskit.quantum_info import SparsePauliOp  # noqa: F401
        from qiskit.circuit import Parameter  # noqa: F401
        from qiskit_aer import AerSimulator  # noqa: F401
        from qiskit_machine_learning.neural_networks import EstimatorQNN  # noqa: F401
        from qiskit_machine_learning.connectors import TorchConnector  # noqa: F401
        return True
    except Exception:
        return False


QUANTUM_AVAILABLE: bool = quantum_available()
