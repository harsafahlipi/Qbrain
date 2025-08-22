
"""
Utility helpers for the Quantum Convolution Block (qConv).
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Tuple
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
# endregion


# region ========== Validation ==========
def require_vector1d(name: str, vec: np.ndarray) -> None:
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if vec.ndim != 1 or vec.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector")


def require_positive_int(name: str, val: int) -> None:
    if not isinstance(val, int) or val <= 0:
        raise ValueError(f"{name} must be a positive integer")
# endregion


# region ========== Math helpers ==========
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalized copy (safe for zero vectors)."""
    norm = float(np.linalg.norm(vec))
    return (vec / norm) if norm != 0.0 else vec


def next_pow2_qubits(n: int) -> int:
    """Return minimum number of qubits to embed n amplitudes (at least 1)."""
    if n <= 1:
        return 1
    import math
    return int(math.ceil(math.log2(n)))


def pad_to_length(vec: np.ndarray, length: int) -> np.ndarray:
    """Right-pad vector with zeros to the desired length."""
    if vec.size > length:
        raise ValueError("pad_to_length: input longer than target length")
    if vec.size == length:
        return vec.astype(float, copy=False)
    return np.pad(vec.astype(float, copy=False), (0, length - vec.size))
# endregion


__all__ = [
    "require_vector1d",
    "require_positive_int",
    "l2_normalize",
    "next_pow2_qubits",
    "pad_to_length",
]
