
"""
QCross Attention (Quantum) — Core Module
=======================================

A Qiskit-based, quantum-inspired Cross-Attention block with clean architecture:
- Docstrings, DRY, SOLID, Type hints, KISS
- Strategy Pattern for feature extraction
- Module-level configuration constants (globals)
- Region-bounded sections for readability
"""

from __future__ import annotations

# region ========== Standard Library & Typing ==========
from typing import Dict, List, Protocol, Tuple, runtime_checkable, Optional
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


# region ========== Core Cross Attention Class ==========
class CrossAttention:
    """Quantum Cross Attention using Qiskit.

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
        """Initialize CrossAttention with configurable dependencies."""
        require_positive("shots", shots)
        self.feature_extractor = feature_extractor or SimpleResizeExtractor(size=size, reduce_dim=DEFAULT_REDUCE_DIM)
        self.shots = shots
        self.size = size
        self.backend = backend or AerSimulator(method=sim_method, max_parallel_threads=max_threads)

    # ---- Feature Extraction Facade ----
    def image_to_feature_vector(self, image_path: str) -> np.ndarray:
        """Facade forwarding to feature_extractor strategy."""
        return self.feature_extractor.image_to_feature_vector(image_path)

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
        """Compute a weighted sum over value vectors using bitstring-indexed scores.

        Notes:
            The bitstrings are interpreted as integer indices; modulo is used if needed.
        """
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
    def quantum_cross_attention(
        self,
        vec_Q: np.ndarray,
        vec_K_list: List[np.ndarray],
        vec_V_list: List[np.ndarray],
    ) -> tuple[np.ndarray, Dict[str, float]]:
        """Run the full Cross Attention pipeline.

        Steps:
            1) Amplitude-encode Q and each K.
            2) For each K, run swap-test(Q, K) to get similarity scores.
            3) Softmax-normalize scores across keys.
            4) Weighted sum over value vectors V using normalized scores.
        """
        # Validate input sizes
        if len(vec_K_list) != len(vec_V_list):
            raise ValueError("vec_K_list and vec_V_list must have the same length.")

        # 1) Encode Q and Ks
        qc_Q = self.amplitude_encode(vec_Q)
        qc_K_list = [self.amplitude_encode(k) for k in vec_K_list]

        # 2) Swap tests per key
        raw_scores: Dict[str, float] = {}
        for i, qc_K in enumerate(qc_K_list):
            swap_circ = self.swap_test(qc_Q, qc_K)
            result = self.measure_state(swap_circ)
            # Use P(0) from ancilla as similarity score; key as zero-padded binary index
            score = result.get("0", 0.0)
            key = format(i, "02b")  # 2-bit label for up to 4 keys; expand as needed
            raw_scores[key] = score

        # 3) Normalize
        attn_weights = self.softmax(raw_scores)

        # 4) Weighted sum of values
        value_matrix = np.stack(vec_V_list, axis=0)  # shape (N, D)
        attn_output = self.weighted_sum_of_values(attn_weights, value_matrix)

        return attn_output, attn_weights
# endregion
