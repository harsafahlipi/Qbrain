
"""
Quantum Slot Attention — core implementation.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from dataclasses import dataclass
from typing import List
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
# endregion

# region ========== Internal Imports ==========
from .circuits import amplitude_encode, swap_test
from .runner import measure_counts
from .math_ops import weighted_sum_of_values
from .utils import softmax_from_dict, DEFAULT_SEED, require_nonempty_vector, require_positive_int
# endregion


@dataclass(frozen=True)
class QuantumSlotAttentionConfig:
    """Configuration for Quantum Slot Attention."""
    num_slots: int = 2
    seed: int = DEFAULT_SEED
    shots: int = 1024


class QuantumSlotAttention:
    """
    Quantum Slot Attention mechanism using swap tests between slots and keys.

    Workflow:
      1) Initialize normalized slot vectors (Gaussian, fixed seed)
      2) Encode slots and key via amplitude encoding
      3) For each slot, run swap test(slot, key) → softmax over counts
      4) Weighted-sum value vectors using the attention scores (placeholder: duplicates of V)
      5) Concatenate slot outputs
    """

    def __init__(self, feature_dim: int, config: QuantumSlotAttentionConfig | None = None) -> None:
        require_positive_int("feature_dim", int(feature_dim))
        self.feature_dim = int(feature_dim)
        self.config = config or QuantumSlotAttentionConfig()

    # region ---- API ----
    def forward(self, vec_K: np.ndarray, vec_V: np.ndarray) -> np.ndarray:
        """Compute slot-attended representation given key/value vectors."""
        require_nonempty_vector("vec_K", vec_K)
        require_nonempty_vector("vec_V", vec_V)

        # Initialize slots
        rng = np.random.default_rng(self.config.seed)
        slots = rng.normal(size=(self.config.num_slots, self.feature_dim)).astype(np.float32)
        slots /= (np.linalg.norm(slots, axis=1, keepdims=True) + 1e-12)

        # Encode key and value
        qc_K = amplitude_encode(vec_K)
        # Value is used classically; we keep its vector form

        slot_outputs: List[np.ndarray] = []
        for s in slots:
            qc_S = amplitude_encode(s)
            swap = swap_test(qc_S, qc_K)
            counts = measure_counts(swap, shots=self.config.shots)
            attn = softmax_from_dict(counts)
            # Placeholder multi-state bank: duplicate V
            value_bank = [vec_V, vec_V]
            slot_outputs.append(weighted_sum_of_values(attn, value_bank))

        return np.concatenate(slot_outputs, axis=0)

    # Convenience alias
    __call__ = forward
    # endregion


def quantum_slot_attention(vec_K: np.ndarray, vec_V: np.ndarray, num_slots: int = 2, feature_dim: int = 16, shots: int = 1024) -> np.ndarray:
    """Functional wrapper replicating the original signature."""
    attn = QuantumSlotAttention(feature_dim=feature_dim, config=QuantumSlotAttentionConfig(num_slots=num_slots, shots=shots))
    return attn.forward(vec_K, vec_V)
