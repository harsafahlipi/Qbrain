
"""
qConv — Aggregator Module
=========================

Re-exports:
- QConvPatchBlockAmp, QConvPatchConfig
- QConvAttention
- quantum_conv_patch_block_amp, attention_block (functional wrappers)
"""

from __future__ import annotations

from .qconv_patch import QConvPatchBlockAmp, QConvPatchConfig
from .attention import QConvAttention, quantum_conv_patch_block_amp, attention_block

__all__ = [
    "QConvPatchBlockAmp",
    "QConvPatchConfig",
    "QConvAttention",
    "quantum_conv_patch_block_amp",
    "attention_block",
]
