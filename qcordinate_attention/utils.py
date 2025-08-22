
"""
Utilities and configuration for the Coordinate Attention module.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

# ---- Module-level configuration ("globals") ----
DEFAULT_SHOTS: int = 1024
DEFAULT_REDUCE_DIM: int = 16
DEFAULT_SIM_METHOD: str = "statevector"
DEFAULT_MAX_THREADS: int = 8
DEFAULT_SIZE: Tuple[int, int] = (4, 4)


def require_positive(name: str, value: int) -> None:
    """Validate positive integer parameters.

    Args:
        name: Parameter name.
        value: Parameter value.

    Raises:
        ValueError: If value <= 0.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector if its norm is nonzero."""
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm != 0 else vec


def pad_to_pow2(vec: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad a 1D vector with zeros to the next power-of-two length.

    Returns:
        (padded_vector, n_qubits)
    """
    vec = np.asarray(vec).ravel()
    length = len(vec)
    if length <= 0:
        raise ValueError("Vector length must be positive for encoding.")
    import math
    n_qubits = int(math.ceil(math.log2(length)))
    padded_len = 2 ** n_qubits
    if length < padded_len:
        import numpy as _np
        vec = _np.pad(vec, (0, padded_len - length))
    return vec, n_qubits
