
"""
Utilities and configuration for the Quantum Slot Attention module.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Dict, Iterable
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
# endregion

# region ========== Globals / Defaults ==========
DEFAULT_IMAGE_SIZE = (4, 4)
DEFAULT_REDUCE_DIM = 16
DEFAULT_SHOTS = 1024
DEFAULT_SEED = 42
# endregion


# region ========== Math helpers ==========
def softmax_from_dict(scores: Dict[str, float]) -> Dict[str, float]:
    """Stable softmax over a dict of scores (values), keyed by bitstrings."""
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=float)
    values -= np.max(values)
    exp = np.exp(values)
    denom = float(np.sum(exp)) or 1e-10
    probs = exp / denom
    return dict(zip(scores.keys(), probs.tolist()))
# endregion


# region ========== Validation helpers ==========
def require_nonempty_vector(name: str, vec: np.ndarray) -> None:
    """Validate that vec is a 1D non-empty numpy array."""
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if vec.ndim != 1 or vec.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector")

def require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
# endregion


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_REDUCE_DIM",
    "DEFAULT_SHOTS",
    "DEFAULT_SEED",
    "softmax_from_dict",
    "require_nonempty_vector",
    "require_positive_int",
]
