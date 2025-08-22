
"""
Configuration objects for qcombined_encoding.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CombinedEncodingConfig:
    """
    Configuration for the combined rotation + amplitude encoding quantum model.

    Attributes:
        n_rot: Number of qubits dedicated to rotation (RY) encoding.
        n_amp: Number of qubits dedicated to amplitude encoding.
        n_layers: Number of StronglyEntanglingLayers repetitions.
    """
    n_rot: int = 2
    n_amp: int = 2
    n_layers: int = 2

    @property
    def n_qubits(self) -> int:
        return int(self.n_rot + self.n_amp)
