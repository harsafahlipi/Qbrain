
"""
Torch modules for qcombined_encoding with PennyLane backend and a pure-PyTorch fallback.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import pennylane as qml  # noqa: F401
    _HAS_QML = True
except Exception:
    _HAS_QML = False

from .config import CombinedEncodingConfig
from .circuits import make_qnode


class QuantumClassifier(nn.Module):
    """
    Quantum classifier head using combined rotation + amplitude encoding (PennyLane).
    Outputs a single logit per sample (PauliZ expectation on wire-0).
    """

    def __init__(self, config: CombinedEncodingConfig) -> None:
        super().__init__()
        self.cfg = config
        self.weights = nn.Parameter(0.01 * torch.randn(self.cfg.n_layers, self.cfg.n_qubits, 3))

        if _HAS_QML:
            self.qnode = make_qnode(self.cfg)
        else:
            self.qnode = None  # Fallback will be used

        # Fallback MLP to approximate behavior when PennyLane is missing
        self._fallback = nn.Sequential(
            nn.Linear(self.cfg.n_rot + (2 ** self.cfg.n_amp), 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward_single(self, x_rot: torch.Tensor, x_amp: torch.Tensor) -> torch.Tensor:
        """
        Forward for a single sample: (x_rot, x_amp) -> scalar logit.
        """
        if self.qnode is not None:
            return self.qnode(x_rot, x_amp, self.weights)  # type: ignore[misc]
        # Fallback: pad and concat, process with MLP
        with torch.no_grad():
            target_len = 2 ** self.cfg.n_amp
            if x_amp.numel() < target_len:
                pad = torch.zeros(target_len - x_amp.numel(), dtype=x_amp.dtype, device=x_amp.device)
                x_amp = torch.cat([x_amp, pad], dim=0)
            x_in = torch.cat([x_rot, x_amp], dim=0)
        return self._fallback(x_in.unsqueeze(0)).squeeze(0).squeeze(-1)

    def forward_batch(self, x_rot: torch.Tensor, x_amp: torch.Tensor) -> torch.Tensor:
        """
        Batched forward using per-sample calls; returns shape (batch,).
        """
        outputs = []
        for i in range(x_rot.size(0)):
            outputs.append(self.forward_single(x_rot[i], x_amp[i]))
        return torch.stack(outputs)
