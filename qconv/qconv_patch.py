
"""
Quantum convolution on one patch using amplitude encoding.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from dataclasses import dataclass
from typing import Sequence, Optional
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
# endregion

# region ========== Internal ==========
from .circuits import amplitude_encoding_circuit, apply_filter_and_entanglement
from .expectation import z_expectations
from .utils import require_vector1d
# endregion


@dataclass(frozen=True)
class QConvPatchConfig:
    """Configuration for a qConv per-patch operation."""
    # reserved for future options (e.g., entanglement style, reps)
    pass


class QConvPatchBlockAmp:
    """
    Single-patch quantum convolution:
      1) amplitude encoding
      2) parameterized RY filter + CNOT chain
      3) readout via ⟨Z_i⟩
    """

    def __init__(self, config: Optional[QConvPatchConfig] = None) -> None:
        self.config = config or QConvPatchConfig()

    def forward(self, input_patch: np.ndarray, weight_params: Sequence[float]) -> np.ndarray:
        require_vector1d("input_patch", input_patch)
        qc = amplitude_encoding_circuit(input_patch)
        qc = apply_filter_and_entanglement(qc, weight_params)
        return z_expectations(qc)

    __call__ = forward
