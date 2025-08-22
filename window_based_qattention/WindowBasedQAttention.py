
"""
window_based_qattention — Aggregator Module
===========================================

Exports:
- WindowBasedQSelfAttention class + WindowAttentionConfig
- window_based_qself_attention functional API
- image_to_feature_vector utility (for examples)
"""

from __future__ import annotations

from .window_based_qself_attention_core import WindowBasedQSelfAttention, WindowAttentionConfig, window_based_qself_attention
from .preprocess import image_to_feature_vector

__all__ = ["WindowBasedQSelfAttention", "WindowAttentionConfig", "window_based_qself_attention", "image_to_feature_vector"]
