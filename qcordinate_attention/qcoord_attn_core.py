
"""
Coordinate Attention (Quantum) — Core Module
============================================

A Qiskit-based, quantum-inspired Coordinate Attention block with clean architecture:
- Docstrings, DRY, SOLID, Type hints, KISS
- Strategy Pattern for feature extraction
- Module-level configuration constants
- Region-bounded sections for readability
"""

from __future__ import annotations

# region ========== Standard Library & Typing ==========
from typing import Dict, Protocol, Tuple, runtime_checkable, Optional
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
import cv2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
# endregion

# region ========== Internal Utilities / Config ==========
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
# endregion


# region ========== Strategy Pattern: Feature Extraction Interfaces ==========
@runtime_checkable
class FeatureExtractor(Protocol):
    """Interface for feature extraction strategies."""

    def image_to_feature_vector(self, image_path: str) -> np.ndarray:
        """Convert an image to a normalized feature vector."""
        ...


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
        """Convert image to normalized feature vector with simple dimensionality cap.

        Args:
            image_path: Path to grayscale or color image.

        Returns:
            Normalized feature vector (np.float32).

        Raises:
            ValueError: If image cannot be read.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image at path: {image_path}")
        resized = cv2.resize(image, self.size, interpolation=self.interpolation)
        flattened = resized.flatten().astype(np.float32)
        if len(flattened) > self.reduce_dim:
            flattened = flattened[: self.reduce_dim]
        return normalize_vector(flattened)
# endregion


# region ========== Core Coordinate Attention Class ==========
class CoordinateAttention:
    """Quantum Coordinate Attention using Qiskit.

    SOLID:
        - SRP: Each method has a single responsibility.
        - OCP: Feature extractor is pluggable; backend is injected.
        - LSP: Any FeatureExtractor satisfying the Protocol works.
        - ISP: Strategy exposes only what's needed.
        - DIP: Depends on abstractions, not concretions.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        shots: int = DEFAULT_SHOTS,
        backend: Optional[AerSimulator] = None,
        sim_method: str = DEFAULT_SIM_METHOD,
        max_threads: int = DEFAULT_MAX_THREADS,
        size: Tuple[int, int] = DEFAULT_SIZE,
    ) -> None:
        """Initialize CoordinateAttention with configurable dependencies."""
        require_positive("shots", shots)
        self.feature_extractor = feature_extractor or SimpleResizeExtractor(size=size, reduce_dim=DEFAULT_REDUCE_DIM)
        self.shots = shots
        self.size = size
        self.backend = backend or AerSimulator(method=sim_method, max_parallel_threads=max_threads)

    # ---- Feature Extraction Facade ----
    def image_to_feature_vector(self, image_path: str) -> np.ndarray:
        """Facade forwarding to feature_extractor strategy."""
        return self.feature_extractor.image_to_feature_vector(image_path)

    # ---- Coordinate Pooling ----
    def coordinate_pooling(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a flattened feature map into horizontal and vertical pooled components.

        Args:
            vector: Flattened vector representing an HxW map.

        Returns:
            (h_pool, v_pool): Means over rows and columns respectively.
        """
        h, w = self.size
        if len(vector) != h * w:
            raise ValueError(f"Vector length must match {h*w} for size={self.size}")
        matrix = vector.reshape(self.size)
        h_pool = matrix.mean(axis=0)  # shape (w,)
        v_pool = matrix.mean(axis=1)  # shape (h,)
        return h_pool.astype(np.float32), v_pool.astype(np.float32)

    # ---- Amplitude Encoding ----
    @staticmethod
    def amplitude_encode(vector: np.ndarray) -> QuantumCircuit:
        """Amplitude-encode a real vector into a quantum state (with padding & normalization)."""
        vec = np.asarray(vector, dtype=np.float64).ravel()
        if vec.size == 0 or np.allclose(vec, 0):
            raise ValueError("Input vector cannot be empty or all zeros for amplitude encoding.")
        vec, n_qubits = pad_to_pow2(vec)
        vec = normalize_vector(vec)
        qc = QuantumCircuit(n_qubits)
        qc.initialize(vec, range(n_qubits))
        return qc

    # ---- Swap Test ----
    @staticmethod
    def swap_test(qc1: QuantumCircuit, qc2: QuantumCircuit) -> QuantumCircuit:
        """Construct a swap-test circuit to estimate similarity between two states.

        Measures the ancilla; probability distribution over {0,1} used as scores.
        """
        if qc1.num_qubits != qc2.num_qubits:
            raise ValueError("Circuits must have the same number of qubits.")
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
        """Transpile & run the circuit; return normalized counts (probabilities)."""
        tqc = transpile(qc, self.backend, optimization_level=3)
        result = self.backend.run(tqc, shots=self.shots).result()
        counts = result.get_counts()
        if not counts:
            return {}
        return {state: count / float(self.shots) for state, count in counts.items()}

    # ---- Math Utils ----
    @staticmethod
    def softmax(scores: Dict[str, float]) -> Dict[str, float]:
        """Softmax over dict values keyed by bitstrings."""
        if not scores:
            return {}
        keys = list(scores.keys())
        values = np.array([scores[k] for k in keys], dtype=np.float64)
        exp = np.exp(values - np.max(values))
        probs = exp / np.sum(exp)
        return dict(zip(keys, probs))

    @staticmethod
    def weighted_sum_of_values(attn_scores: Dict[str, float], value_vectors: np.ndarray) -> np.ndarray:
        """Compute a weighted sum over value vectors using bitstring-indexed scores."""
        if value_vectors.ndim != 2:
            raise ValueError("value_vectors must be a 2D array of shape (N, D).")
        out = np.zeros_like(value_vectors[0], dtype=np.float64)
        n = len(value_vectors)
        for state, score in attn_scores.items():
            try:
                idx = int(state, 2) % n
                out += score * value_vectors[idx]
            except ValueError:
                continue
        return out.astype(np.float32)

    # ---- Full Pipeline ----
    def quantum_coordinate_attention(self, vec_Q: np.ndarray, vec_K: np.ndarray, vec_V: np.ndarray) -> np.ndarray:
        """Run the full Coordinate Attention pipeline on Q/K/V vectors.

        Steps:
            1) Coordinate pooling (horizontal/vertical) for Q, K, V.
            2) Amplitude-encode each pooled vector.
            3) Swap tests for horizontal and vertical attention (Q vs K).
            4) Measure and softmax-normalize scores separately.
            5) Combine scores (average) and compute weighted sums over V-pools.
            6) Concatenate horizontal and vertical outputs.
        """
        # 1) Coordinate pooling
        h_Q, v_Q = self.coordinate_pooling(vec_Q)
        h_K, v_K = self.coordinate_pooling(vec_K)
        h_V, v_V = self.coordinate_pooling(vec_V)

        # 2) Encode
        qc_Q_h, qc_K_h, qc_V_h = map(self.amplitude_encode, (h_Q, h_K, h_V))
        qc_Q_v, qc_K_v, qc_V_v = map(self.amplitude_encode, (v_Q, v_K, v_V))

        # 3) Swap tests
        swap_circuit_h = self.swap_test(qc_Q_h, qc_K_h)
        swap_circuit_v = self.swap_test(qc_Q_v, qc_K_v)

        # 4) Measure + softmax
        scores_h = self.softmax(self.measure_state(swap_circuit_h))
        scores_v = self.softmax(self.measure_state(swap_circuit_v))

        # 5) Combine (average) and weighted sums over duplicated V-pools (placeholder)
        combined_states = set(scores_h) | set(scores_v)
        combined_scores = {s: (scores_h.get(s, 0.0) + scores_v.get(s, 0.0)) / 2.0 for s in combined_states}

        value_vectors_h = np.stack([h_V, h_V], axis=0)
        value_vectors_v = np.stack([v_V, v_V], axis=0)

        out_h = self.weighted_sum_of_values(combined_scores, value_vectors_h)
        out_v = self.weighted_sum_of_values(combined_scores, value_vectors_v)

        # 6) Concatenate
        return np.concatenate([out_h, out_v]).astype(np.float32)
# endregion
