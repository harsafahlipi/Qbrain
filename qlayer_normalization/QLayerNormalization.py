
"""
QLayerNormalization — Aggregator Module
=======================================

Brings together the two layer-norm variants under a single import path.
"""

from __future__ import annotations

from .quantum_layer_normalization import QuantumLayerNormalization
from .variational_layer_norm import VariationalQuantumLayerNorm

__all__ = ["QuantumLayerNormalization", "VariationalQuantumLayerNorm"]
