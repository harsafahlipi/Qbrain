
"""
QRAM-enhanced QSANN stack: multiple QRAM_Attention layers with layer norm.
"""

from __future__ import annotations

# region ========== Third-Party ==========
import torch.nn as nn
# endregion

# region ========== Local Imports ==========
from .qram_attention import QRAM_Attention
# endregion


class QRAM_QSANN_Working(nn.Module):
    """A simple QSANN using QRAM-enhanced attention layers."""

    def __init__(self, embed_dim: int = 64, num_layers: int = 2, num_heads: int = 1, qram_size: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([QRAM_Attention(embed_dim, num_heads, qram_size) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

    def forward(self, x):
        for attn, ln in zip(self.layers, self.norms):
            x = ln(attn(x))
        return x
