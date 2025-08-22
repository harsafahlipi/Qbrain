
"""
QRAM-enhanced attention block that uses QRAM_Layer for Q, K, V projections.
"""

from __future__ import annotations

# region ========== Stdlib / Typing ==========
import math
# endregion

# region ========== Third-Party ==========
import torch
import torch.nn as nn
# endregion

# region ========== Local Imports ==========
from .qram_layer import QRAM_Layer
# endregion


class QRAM_Attention(nn.Module):
    """Self-attention with QRAM-based projections for Q, K, and V, plus residual path."""

    def __init__(self, embed_dim: int, num_heads: int = 1, qram_size: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // max(1, num_heads)

        self.q_proj = QRAM_Layer(embed_dim, embed_dim, qram_size=qram_size)
        self.k_proj = QRAM_Layer(embed_dim, embed_dim, qram_size=qram_size)
        self.v_proj = QRAM_Layer(embed_dim, embed_dim, qram_size=qram_size)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention: softmax(QK^T / sqrt(d)) V with residual connection."""
        b, t, d = x.shape
        x_flat = x.view(b * t, d)

        Q = self.q_proj(x_flat).view(b, t, d)
        K = self.k_proj(x_flat).view(b, t, d)
        V = self.v_proj(x_flat).view(b, t, d)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = self.out_proj(out)
        return out + x
