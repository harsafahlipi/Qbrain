
"""
Quantum circuit factory for combined rotation + amplitude encoding.
"""
from __future__ import annotations
from typing import Tuple

import torch

try:
    import pennylane as qml
    _HAS_QML = True
except Exception:
    qml = None  # type: ignore
    _HAS_QML = False

from .config import CombinedEncodingConfig


def _rotation_encoding(x_rot: torch.Tensor, wires: range) -> None:
    """
    Apply RY rotations for rotation-encoded subset.
    """
    for i, w in enumerate(wires):
        qml.RY(x_rot[i], wires=w)  # type: ignore[attr-defined]


def _amp_embed_from_tensor(x_amp: torch.Tensor, start_wire: int, n_amp: int) -> None:
    """
    Amplitude-embed x_amp across n_amp wires starting at start_wire.
    Pads to 2**n_amp and normalizes inside the qnode (normalize=False).
    """
    target_len = 2 ** n_amp
    if x_amp.numel() < target_len:
        pad = torch.zeros(target_len - x_amp.numel(), dtype=x_amp.dtype, device=x_amp.device)
        x_amp = torch.cat([x_amp, pad], dim=0)
    norm = torch.linalg.vector_norm(x_amp)
    x_amp = x_amp / (norm + 1e-12)
    qml.AmplitudeEmbedding(x_amp, wires=range(start_wire, start_wire + n_amp), normalize=False)  # type: ignore[attr-defined]


def make_qnode(cfg: CombinedEncodingConfig, dev=None):
    """
    Build a PennyLane QNode for the combined encoding with StronglyEntanglingLayers.
    Returns a callable (x_rot: Tensor, x_amp: Tensor, weights: Parameter) -> Tensor[()] (expectation).
    """
    if not _HAS_QML:
        raise RuntimeError("PennyLane is required to build the quantum node.")

    n_qubits = cfg.n_qubits
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)  # type: ignore[attr-defined]

    @qml.qnode(dev, interface="torch")  # type: ignore[misc]
    def quantum_net(x_rot: torch.Tensor, x_amp: torch.Tensor, weights: torch.Tensor):
        # Rotation encoding on the first n_rot wires
        if cfg.n_rot > 0:
            _rotation_encoding(x_rot, range(cfg.n_rot))
        # Amplitude encoding on the remaining wires
        if cfg.n_amp > 0:
            _amp_embed_from_tensor(x_amp, start_wire=cfg.n_rot, n_amp=cfg.n_amp)
            # Light cross-connection between encoding groups
            for i in range(cfg.n_amp):
                qml.CNOT(wires=[i % max(1, cfg.n_rot), cfg.n_rot + i])  # type: ignore[attr-defined]

        # Variational block
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))  # type: ignore[attr-defined]
        return qml.expval(qml.PauliZ(0))  # type: ignore[attr-defined]

    return quantum_net
