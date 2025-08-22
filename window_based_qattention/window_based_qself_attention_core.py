
"""
Window-based Quantum Self-Attention — core implementation.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from dataclasses import dataclass
from typing import List
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
# endregion

# region ========== Internal ==========
from .windows import split_into_windows
from .circuits import amplitude_encode, swap_test
from .runner import measure_counts
from .math_ops import weighted_sum_of_values
from .utils import stable_softmax_from_dict, require_vector1d, require_positive_int
# endregion


@dataclass(frozen=True)
class WindowAttentionConfig:
    """Configuration for window-based Q self-attention."""
    window_size: int = 4
    shots: int = 1024


class WindowBasedQSelfAttention:
    """
    Split Q/K/V into fixed-size windows, run swap-test attention per-window,
    and concatenate the attention-weighted outputs.
    """

    def __init__(self, config: WindowAttentionConfig | None = None) -> None:
        self.config = config or WindowAttentionConfig()

    # region ---- API ----
    def forward(self, vec_Q: np.ndarray, vec_K: np.ndarray, vec_V: np.ndarray) -> np.ndarray:
        """Compute windowed quantum self-attention over (Q, K, V)."""
        require_vector1d("vec_Q", vec_Q)
        require_vector1d("vec_K", vec_K)
        require_vector1d("vec_V", vec_V)
        require_positive_int("window_size", self.config.window_size)

        wQ = split_into_windows(vec_Q, self.config.window_size)
        wK = split_into_windows(vec_K, self.config.window_size)
        wV = split_into_windows(vec_V, self.config.window_size)
        if not (len(wQ) == len(wK) == len(wV)):
            raise ValueError("All vectors must yield the same number of windows.")

        outputs: List[np.ndarray] = []
        for q, k, v in zip(wQ, wK, wV):
            qc_Q = amplitude_encode(q)
            qc_K = amplitude_encode(k)
            swap = swap_test(qc_Q, qc_K)
            raw = measure_counts(swap, shots=self.config.shots)
            attn = stable_softmax_from_dict(raw)
            # Placeholder value-bank
            bank = [v, v]
            outputs.append(weighted_sum_of_values(attn, bank))

        return np.concatenate(outputs, axis=0)

    __call__ = forward
    # endregion


def window_based_qself_attention(vec_Q: np.ndarray, vec_K: np.ndarray, vec_V: np.ndarray, window_size: int = 4, shots: int = 1024) -> np.ndarray:
    """Functional wrapper to match the original signature (without visualization)."""
    model = WindowBasedQSelfAttention(config=WindowAttentionConfig(window_size=window_size, shots=shots))
    return model.forward(vec_Q, vec_K, vec_V)
