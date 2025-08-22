
"""
QuantumTransformerEncoderBlock — attention + FFN with residual/LayerNorm.
"""

from __future__ import annotations
import torch.nn as nn

from .quantum_attention import QuantumAttention


class QuantumTransformerEncoderBlock(nn.Module):
    """Encoder block with quantum attention and classical FFN."""

    def __init__(self, num_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.attention = QuantumAttention(num_qubits, depth)
        self.norm1 = nn.LayerNorm(num_qubits)
        self.norm2 = nn.LayerNorm(num_qubits)
        self.ffn = nn.Sequential(
            nn.Linear(num_qubits, num_qubits * 2),
            nn.ReLU(),
            nn.Linear(num_qubits * 2, num_qubits),
        )

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
