
"""
Qpatch_merging_spliting — Core
==============================

Builds a patch merging/splitting quantum block:
- Per-patch ZZFeatureMap encodings
- Global RealAmplitudes ansatz
- Returns feature map circuit, ansatz circuit, input/weight ParameterVectors, and a composed "full" circuit.

Design principles:
- English docstrings, type hints, DRY/KISS/SOLID
- No prints/side-effects
- Region markers for readability
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Tuple
# endregion

# region ========== Third-Party Imports ==========
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
# endregion

# region ========== Internal Utilities / Config ==========
from .utils import DEFAULT_ANSATZ_REPS, DEFAULT_FM_REPS, require_positive, require_equal
# endregion


class QuantumPatchMergingSplitingBlock:
    """
    Quantum patch merging/splitting block.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits. Must equal patch_size * num_patches.
    patch_size : int
        Number of features per patch.
    num_patches : int
        Number of patches.
    ansatz_reps : int, default=DEFAULT_ANSATZ_REPS
        Repetitions for the RealAmplitudes ansatz.
    feature_map_reps : int, default=DEFAULT_FM_REPS
        Repetitions for per-patch ZZFeatureMap (kept 1 to match original API).
    """

    def __init__(
        self,
        num_qubits: int,
        patch_size: int,
        num_patches: int,
        ansatz_reps: int = DEFAULT_ANSATZ_REPS,
        feature_map_reps: int = DEFAULT_FM_REPS,
    ) -> None:
        require_positive("num_qubits", num_qubits)
        require_positive("patch_size", patch_size)
        require_positive("num_patches", num_patches)
        require_positive("ansatz_reps", ansatz_reps)
        require_positive("feature_map_reps", feature_map_reps)

        require_equal("num_qubits", num_qubits, "patch_size * num_patches", patch_size * num_patches)

        self.num_qubits = num_qubits
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.ansatz_reps = ansatz_reps
        self.feature_map_reps = feature_map_reps

    # region ---- Build helpers ----
    def _build_feature_map(self, qr: QuantumRegister) -> Tuple[QuantumCircuit, ParameterVector]:
        """Build the per-patch ZZFeatureMap encoding circuit and its input ParameterVector."""
        num_qubits = self.num_qubits
        patch_size = self.patch_size
        num_patches = self.num_patches
        reps = self.feature_map_reps

        feature_map_circuit = QuantumCircuit(qr)
        # Input parameters for all patches (length = patch_size * num_patches)
        input_params = ParameterVector("x", patch_size * num_patches)

        for patch in range(num_patches):
            start = patch * patch_size
            stop = start + patch_size
            patch_params = input_params[start:stop]

            fmap = ZZFeatureMap(
                feature_dimension=patch_size,
                reps=reps,
                entanglement="linear",
                parameter_prefix=f"x_{patch}",
            )

            # For reps=1, fmap.num_parameters == patch_size. We keep this strict to match original code.
            if len(patch_params) != fmap.num_parameters:
                raise ValueError(
                    f"Parameter mismatch for patch {patch}: "
                    f"{len(patch_params)} provided vs {fmap.num_parameters} required."
                )

            # Map this patch's feature map onto its contiguous qubit range
            qubit_slice = list(range(patch * patch_size, (patch + 1) * patch_size))
            feature_map_circuit.compose(fmap.assign_parameters(patch_params), qubits=qubit_slice, inplace=True)

        return feature_map_circuit, input_params

    def _build_ansatz(self) -> Tuple[QuantumCircuit, ParameterVector]:
        """Build the global RealAmplitudes ansatz and its weight ParameterVector."""
        ansatz = RealAmplitudes(self.num_qubits, reps=self.ansatz_reps, parameter_prefix="w")
        weight_params = ParameterVector("w", ansatz.num_parameters)
        if len(weight_params) != ansatz.num_parameters:
            raise ValueError(
                f"Parameter mismatch for ansatz: {len(weight_params)} provided vs {ansatz.num_parameters} required."
            )
        ansatz_circuit = ansatz.assign_parameters(weight_params)
        return ansatz_circuit, weight_params
    # endregion

    # region ---- Public API ----
    def build(self) -> Tuple[QuantumCircuit, QuantumCircuit, ParameterVector, ParameterVector, QuantumCircuit]:
        """
        Build and return (feature_map_circuit, ansatz_circuit, input_params, weight_params, full_circuit).
        """
        qr = QuantumRegister(self.num_qubits, "q")

        # Feature map and inputs
        fmap_circ, input_params = self._build_feature_map(qr)

        # Ansatz and weights
        ansatz_circ, weight_params = self._build_ansatz()

        # Compose the full circuit (no prints; measurement optional for debugging)
        full = QuantumCircuit(qr)
        full.compose(fmap_circ, qubits=range(self.num_qubits), inplace=True)
        full.compose(ansatz_circ, qubits=range(self.num_qubits), inplace=True)
        # Note: leave measurement to the caller; keeping the circuit pure here.

        return fmap_circ, ansatz_circ, input_params, weight_params, full
    # endregion


def patch_merging_block(
    num_qubits: int,
    patch_size: int,
    num_patches: int,
    reps: int = DEFAULT_ANSATZ_REPS,
) -> Tuple[QuantumCircuit, QuantumCircuit, ParameterVector, ParameterVector, QuantumCircuit]:
    """
    Convenience function mirroring the original API (reps -> ansatz reps).

    Returns
    -------
    (feature_map_circuit, ansatz_circuit, input_params, weight_params, full_circuit)
    """
    block = QuantumPatchMergingSplitingBlock(
        num_qubits=num_qubits,
        patch_size=patch_size,
        num_patches=num_patches,
        ansatz_reps=reps,
        feature_map_reps=1,  # fixed to match original snippet
    )
    return block.build()
