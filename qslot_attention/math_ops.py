
"""
Math helpers for Slot Attention.
"""

from __future__ import annotations
from typing import Dict, Sequence
import numpy as np


def weighted_sum_of_values(attn_scores: Dict[str, float], value_vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Compute weighted sum of given value vectors using bitstring-keyed scores."""
    if not value_vectors:
        raise ValueError("value_vectors must be non-empty")
    out = np.zeros_like(value_vectors[0])
    k = len(value_vectors)
    for state, score in attn_scores.items():
        try:
            idx = int(state, 2) % k
            out = out + float(score) * value_vectors[idx]
        except ValueError:
            continue
    return out
