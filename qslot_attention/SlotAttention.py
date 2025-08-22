
"""
qslot_attention — Aggregator Module
==================================

Re-exports:
- QuantumSlotAttention class and SlotAttentionConfig
- quantum_slot_attention convenience function
- image_to_feature_vector for minimal examples
"""

from __future__ import annotations

from .qslot_attention_core import QuantumSlotAttention, SlotAttentionConfig, quantum_slot_attention
from .preprocess import image_to_feature_vector

__all__ = ["QuantumSlotAttention", "SlotAttentionConfig", "quantum_slot_attention", "image_to_feature_vector"]
