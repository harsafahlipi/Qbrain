
"""
Utilities for window-based quantum self-attention.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Dict, List
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
# endregion

# region ========== Globals / Defaults ==========
DEFAULT_SHOTS: int = 1024
# endregion


# region ========== Validation helpers ==========
def require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def require_vector1d(name: str, vec: np.ndarray) -> None:
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if vec.ndim != 1 or vec.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector")
# endregion


# region ========== Math helpers ==========
def stable_softmax_from_dict(scores: Dict[str, float]) -> Dict[str, float]:
    """Numerically stable softmax over a dict of bitstring->score."""
    if not scores:
        return {}
    vals = np.asarray(list(scores.values()), dtype=float)
    vals = vals - np.max(vals)
    exp = np.exp(vals)
    denom = float(np.sum(exp)) or 1e-10
    probs = exp / denom
    return dict(zip(scores.keys(), probs.tolist()))
# endregion


__all__ = [
    "DEFAULT_SHOTS",
    "require_positive_int",
    "require_vector1d",
    "stable_softmax_from_dict",
]
