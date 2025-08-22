
"""
QMasked Self-Attention (Quantum) — Core Module
=============================================

A Qiskit-based, quantum-inspired Masked Self-Attention block with clean architecture:
- Docstrings, DRY, SOLID, Type hints, KISS
- Strategy Pattern for feature extraction
- Module-level configuration constants (globals)
- Region-bounded sections for readability
"""

from __future__ import annotations
from typing import Dict, Protocol, Tuple, runtime_checkable, Optional
import numpy as np
import cv2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from .utils import (
    DEFAULT_SHOTS,
    DEFAULT_REDUCE_DIM,
    DEFAULT_SIM_METHOD,
    DEFAULT_MAX_THREADS,
    DEFAULT_SIZE,
    require_positive,
    normalize_vector,
    pad_to_pow2,
)


# region ========== Strategy Pattern: Feature Extraction Interfaces ==========
@runtime_checkable
class FeatureExtractor(Protocol):
    """Interface for feature extraction strategies."""
    def image_to_feature_vector(self, image_path: str) -> np.ndarray: ...


class SimpleResizeExtractor:
    """Simple feature extractor: resize -> flatten -> slice -> normalize (KISS)."""

    def __init__(
        self,
        size: Tuple[int, int] = DEFAULT_SIZE,
        reduce_dim: int = DEFAULT_REDUCE_DIM,
        interpolation: int = cv2.INTER_AREA,
    ) -> None:
        require_positive("reduce_dim", reduce_dim)
        self.size = size
        self.reduce_dim = reduce_dim
        self.interpolation = interpolation

    def image_to_feature_vector(self, image_path: str) -> np.ndarray:
        """Convert image to normalized feature vector with simple dimensionality cap."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image at path: {image_path}")
        resized = cv2.resize(image, self.size, interpolation=self.interpolation)
        flattened = resized.flatten().astype(np.float32)
        if len(flattened) > self.reduce_dim:
            flattened = flattened[: self.reduce_dim]
        return normalize_vector(flattened)
# endregion


# region ========== Core Masked Self-Attention Class ==========
class MaskedSelfAttention:
    """Quantum Masked Self-Attention using Qiskit."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        shots: int = DEFAULT_SHOTS,
        backend: Optional[AerSimulator] = None,
        sim_method: str = DEFAULT_SIM_METHOD,
        max_threads: int = DEFAULT_MAX_THREADS,
        size: int = 16,
    ) -> None:
        require_positive("shots", shots)
        self.feature_extractor = feature_extractor or SimpleResizeExtractor()
        self.shots = shots
        self.size = size
        self.backend = backend or AerSimulator(method=sim_method, max_parallel_threads=max_threads)

    # ---- Feature Extraction ----
    def image_to_feature_vector(self, image_path: str) -> np.ndarray:
        return self.feature_extractor.image_to_feature_vector(image_path)

    # ---- Mask ----
    @staticmethod
    def create_causal_mask(size: int) -> np.ndarray:
        """Create a lower-triangular (causal) mask."""
        return np.tril(np.ones((size, size), dtype=np.float32))

    # ---- Amplitude Encoding ----
    @staticmethod
    def amplitude_encode(vector: np.ndarray) -> QuantumCircuit:
        vec = np.asarray(vector, dtype=np.float64).ravel()
        if vec.size == 0 or np.allclose(vec, 0):
            raise ValueError("Input vector cannot be empty or zero.")
        vec, n_qubits = pad_to_pow2(vec)
        vec = normalize_vector(vec)
        qc = QuantumCircuit(n_qubits)
        qc.initialize(vec, range(n_qubits))
        return qc

    # ---- Swap Test ----
    @staticmethod
    def swap_test(qc1: QuantumCircuit, qc2: QuantumCircuit) -> QuantumCircuit:
        if qc1.num_qubits != qc2.num_qubits:
            raise ValueError("Circuits must have same number of qubits.")
        n = qc1.num_qubits
        qc = QuantumCircuit(2 * n + 1, 1)
        qc.compose(qc1, qubits=range(1, n + 1), inplace=True)
        qc.compose(qc2, qubits=range(n + 1, 2 * n + 1), inplace=True)
        qc.h(0)
        for i in range(n):
            qc.cswap(0, i + 1, n + i + 1)
        qc.h(0)
        qc.measure(0, 0)
        return qc

    # ---- Execution / Readout ----
    def measure_state(self, qc: QuantumCircuit) -> Dict[str, float]:
        tqc = transpile(qc, self.backend, optimization_level=3)
        result = self.backend.run(tqc, shots=self.shots).result()
        counts = result.get_counts()
        if not counts:
            return {}
        return {state: count / float(self.shots) for state, count in counts.items()}

    # ---- Softmax with Mask ----
    @staticmethod
    def softmax_with_mask(scores: Dict[str, float], mask: np.ndarray) -> Dict[str, float]:
        values = np.array(list(scores.values()))
        exp = np.exp(values - np.max(values))
        masked_exp = exp * mask.flatten()[: len(exp)]
        sum_exp = np.sum(masked_exp)
        if sum_exp == 0:
            sum_exp = 1e-10
        normalized_scores = masked_exp / sum_exp
        return dict(zip(scores.keys(), normalized_scores))

    # ---- Full Pipeline ----
    def quantum_masked_self_attention(
        self,
        vec_Q: np.ndarray,
        vec_K: np.ndarray,
        vec_V: np.ndarray,
    ) -> np.ndarray:
        """Run masked self-attention pipeline."""
        mask = self.create_causal_mask(self.size)
        qc_Q = self.amplitude_encode(vec_Q)
        qc_K = self.amplitude_encode(vec_K)
        qc_V = self.amplitude_encode(vec_V)
        swap_circuit = self.swap_test(qc_Q, qc_K)
        raw_scores = self.measure_state(swap_circuit)
        normalized_scores = self.softmax_with_mask(raw_scores, mask)
        value_vectors = np.stack([vec_V, vec_V], axis=0)
        output_vector = np.zeros_like(vec_V)
        for state, score in normalized_scores.items():
            try:
                index = int(state, 2) % len(value_vectors)
                output_vector += score * value_vectors[index]
            except ValueError:
                continue
        return output_vector.astype(np.float32)
# endregion
