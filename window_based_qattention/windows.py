
"""
Window utilities.
"""

from __future__ import annotations
from typing import List
import numpy as np
from .utils import require_positive_int, require_vector1d


def split_into_windows(vector: np.ndarray, window_size: int) -> List[np.ndarray]:
    """Split a 1D vector into contiguous windows of size window_size (last may be shorter)."""
    require_vector1d("vector", vector)
    require_positive_int("window_size", window_size)
    return [vector[i:i + window_size] for i in range(0, vector.size, window_size)]
