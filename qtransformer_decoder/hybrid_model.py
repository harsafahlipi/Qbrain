
"""
Hybrid quantum-classical model using the decoder QNN followed by a small MLP.
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn

# TorchConnector (optional)
try:
    from qiskit_machine_learning.connectors import TorchConnector  # type: ignore
except Exception:  # pragma: no cover
    TorchConnector = None  # type: ignore


class HybridModel(nn.Module):
    """A simple hybrid classifier: QNN → Linear → ReLU → Linear."""

    def __init__(self, qnn: Any, num_classes: int = 10) -> None:
        super().__init__()
        if TorchConnector is None:
            raise RuntimeError("TorchConnector is not available. Install qiskit-machine-learning.")
        self.qnn = TorchConnector(qnn)
        # Determine output size from the qnn connector
        out_dim = self.qnn.output_shape[0] if hasattr(self.qnn, "output_shape") else 1
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qnn(x)  # (batch, qnn_out)
        x = self.fc(x)
        return x
