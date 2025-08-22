
"""
Utilities and configuration for the QPE (Quantum Positional Encoding) module.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

# ---- Module-level configuration ("globals") ----
DEFAULT_MAX_SEQ_LEN: int = 100
DEFAULT_D_MODEL: int = 4            # Must be a power of two for amplitude encoding
DEFAULT_BASE: int = 10000
DEFAULT_SIM_METHOD: str = "statevector"
DEFAULT_MAX_THREADS: int = 8


def is_power_of_two(n: int) -> bool:
    """Return True if n is a power of two and > 0."""
    return (n > 0) and (n & (n - 1) == 0)


def require_power_of_two(name: str, value: int) -> None:
    """Validate that an integer is a power of two (and > 0)."""
    if not is_power_of_two(value):
        raise ValueError(f"{name} must be a power of two, got {value}.")


def require_positive(name: str, value: int) -> None:
    """Validate positive integer parameters."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector if its norm is nonzero."""
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm != 0 else vec
