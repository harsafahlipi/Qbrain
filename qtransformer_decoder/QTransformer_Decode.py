
"""
QTransformer_Decode — Aggregator Module
======================================

Re-exports the circuit builder, QNN factory, and the HybridModel.
"""

from __future__ import annotations

from .decoder_block import create_quantum_decoder_block
from .qnn_builder import create_qnn
from .hybrid_model import HybridModel

__all__ = ["create_quantum_decoder_block", "create_qnn", "HybridModel"]
