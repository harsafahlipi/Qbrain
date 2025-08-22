
"""
QuantumTransformerModel — minimal image → encoder → classifier pipeline.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from .encoder_block import QuantumTransformerEncoderBlock


class QuantumTransformerModel(nn.Module):
    """
    Simple quantum-enhanced Transformer encoder for 28×28 grayscale images.

    Pipeline:
        flatten → linear embedding (to num_qubits) → [EncoderBlock × L] → classifier
    """

    def __init__(self, num_qubits: int = 4, num_classes: int = 10, num_layers: int = 1, depth: int = 1) -> None:
        super().__init__()
        self.num_qubits = int(num_qubits)

        self.embedding = nn.Linear(28 * 28, self.num_qubits)
        self.encoder_layers = nn.ModuleList([
            QuantumTransformerEncoderBlock(self.num_qubits, depth) for _ in range(int(num_layers))
        ])
        self.classifier = nn.Linear(self.num_qubits, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = x.view(b, -1)           # (b, 784)
        x = self.embedding(x)       # (b, num_qubits)
        x = x.unsqueeze(1)          # (b, 1, num_qubits)
        for layer in self.encoder_layers:
            x = layer(x)            # (b, 1, num_qubits)
        x = x.squeeze(1)            # (b, num_qubits)
        x = self.classifier(x)      # (b, num_classes)
        return x
