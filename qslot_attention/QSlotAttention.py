
"""
qslot_attention — Aggregator Module
==================================

Re-exports:
- QuantumSlotAttention class and QuantumSlotAttentionConfig
- quantum_slot_attention convenience function
- image_to_feature_vector for minimal examples
"""

from __future__ import annotations

from .qslot_attention_core import QuantumSlotAttention, QuantumSlotAttentionConfig, quantum_slot_attention
from .preprocess import image_to_feature_vector

__all__ = ["QuantumSlotAttention", "QuantumSlotAttentionConfig", "quantum_slot_attention", "image_to_feature_vector"]
