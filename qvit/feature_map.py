
"""
QViT patch embedding feature map builder.

Builds a circuit with:
- 1 CLS token qubit (Hadamard)
- n_patches blocks of ZZFeatureMap on 2 qubits per patch (default), parameterized as x_{i}_{j}.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Dict, List, Tuple
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
# endregion


class QViTPatchFeatureMap:
    """
    Patch embedding circuit: CLS + per-patch ZZFeatureMap(2 qubits).
    """

    def __init__(self, n_patches: int = 4, n_qubits_per_patch: int = 2) -> None:
        if n_qubits_per_patch <= 0:
            raise ValueError("n_qubits_per_patch must be positive.")
        if n_patches <= 0:
            raise ValueError("n_patches must be positive.")
        self.n_patches = int(n_patches)
        self.n_qubits_per_patch = int(n_qubits_per_patch)
        self.n_qubits = 1 + self.n_patches * self.n_qubits_per_patch  # +1 for CLS
        # Pre-create parameter handles for deterministic mapping
        self._params: List[Parameter] = [
            Parameter(f"x_{i}_{j}") for i in range(self.n_patches) for j in range(self.n_qubits_per_patch)
        ]

    # region ---- Build / Map ----
    def build(self) -> Tuple[QuantumCircuit, List[Parameter]]:
        """
        Construct the feature map circuit and return it with the ordered parameter list.
        Returns:
            (qc, params) where params are [x_0_0, x_0_1, ..., x_{P-1}_{Q-1}]
        """
        qc = QuantumCircuit(self.n_qubits)
        # CLS token
        qc.h(0)
        # Per-patch encoding
        fmap = ZZFeatureMap(feature_dimension=self.n_qubits_per_patch, reps=1)
        for i in range(self.n_patches):
            # Parameters for this patch
            patch_params = [Parameter(f"x_{i}_{j}") for j in range(self.n_qubits_per_patch)]
            patch_circuit = fmap.assign_parameters(patch_params)
            # Target qubits for this patch (skip CLS at 0)
            qubits = [1 + i * self.n_qubits_per_patch + j for j in range(self.n_qubits_per_patch)]
            qc.compose(patch_circuit, qubits=qubits, inplace=True)
        return qc, list(self._params)

    def map_features_to_parameters(self, x_flat: np.ndarray) -> Dict[Parameter, float]:
        """
        Map a flattened feature vector to the parameter objects.
        Args:
            x_flat: shape (n_patches * n_qubits_per_patch,)
        """
        expected = self.n_patches * self.n_qubits_per_patch
        if x_flat.shape[0] != expected:
            raise ValueError(f"x_flat must have length {expected}, got {x_flat.shape[0]}")
        return {param: float(val) for param, val in zip(self._params, x_flat.tolist())}
    # endregion
