
"""
qConv attention-like wrapper that applies the quantum patch block to a list of patches
and concatenates the expectations.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Iterable, List, Optional, Sequence
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
# endregion

# region ========== Internal ==========
from .qconv_patch import QConvPatchBlockAmp, QConvPatchConfig
from .utils import require_vector1d, next_pow2_qubits
# endregion


def _default_weights_for_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create a default weight vector sized to the number of qubits for the patch."""
    n_qubits = max(1, next_pow2_qubits(patch.size))
    return rng.normal(loc=0.0, scale=1.0, size=(n_qubits,)).astype(float)


class QConvAttention:
    """
    Apply QConvPatchBlockAmp over a sequence of patches and concatenate outputs.
    """

    def __init__(self, seed: Optional[int] = None, patch_config: Optional[QConvPatchConfig] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.block = QConvPatchBlockAmp(patch_config)

    def forward(self, input_patches: Iterable[np.ndarray], weight_params: Optional[Iterable[Sequence[float]]] = None) -> np.ndarray:
        outputs: List[np.ndarray] = []
        weights_list = list(weight_params) if weight_params is not None else None
        for idx, patch in enumerate(input_patches):
            patch = np.asarray(patch, dtype=float)
            require_vector1d(f"input_patches[{idx}]", patch)
            w = (weights_list[idx] if weights_list is not None else _default_weights_for_patch(patch, self.rng))
            out = self.block(patch, w)
            outputs.append(out.astype(float, copy=False))
        return np.concatenate(outputs, axis=0)

    __call__ = forward


# Functional APIs matching the original signatures
def quantum_conv_patch_block_amp(input_patch: np.ndarray, weight_params: Sequence[float]) -> np.ndarray:
    """Functional single-patch qConv (compat wrapper)."""
    return QConvPatchBlockAmp()(np.asarray(input_patch, dtype=float), list(weight_params))


def attention_block(input_patch: Iterable[np.ndarray], weight_params: Optional[Iterable[Sequence[float]]] = None) -> np.ndarray:
    """
    Functional attention wrapper.
    - input_patch: iterable of 1D patch arrays of possibly different sizes
    - weight_params: iterable of weight vectors per patch; if None, random defaults
    """
    return QConvAttention()(list(input_patch), list(weight_params) if weight_params is not None else None)
