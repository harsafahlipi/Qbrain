
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
    make_causal_mask,
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


# region ========== Core Masked Self-Attention Class ==========
class MaskedSelfAttention:
    """Quantum Masked Self-Attention using Qiskit.

    Notes:
        This simplified variant mirrors a single Q-K similarity via swap test,
        then applies a 1D causal mask to the attention logits before softmax.
        For a full sequence-aware self-attention, extend the API to accept
        lists of Q/K/V vectors and build per-position masked scores.
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
        """Initialize MaskedSelfAttention with configurable dependencies."""
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
    def softmax_with_mask(scores: Dict[str, float], mask: np.ndarray) -> Dict[str, float]:
        """Apply softmax to scores after elementwise mask.

        Args:
            scores: Dict mapping bitstrings to scores.
            mask: A vector or matrix mask. If matrix, it will be flattened and
                  truncated/expanded (with zeros) to match the number of scores.

        Returns:
            Dict with normalized (masked) probabilities.
        """
        if not scores:
            return {}
        keys = list(scores.keys())
        values = np.array([scores[k] for k in keys], dtype=np.float64)

        # Prepare a 1D mask aligned to values
        m = mask
        if m.ndim == 2:
            m = m.flatten()
        if m.size < values.size:
            pad = np.zeros(values.size - m.size, dtype=m.dtype)
            m = np.concatenate([m, pad])
        elif m.size > values.size:
            m = m[: values.size]

        exp = np.exp(values - np.max(values))
        masked_exp = exp * m
        denom = masked_exp.sum()
        if denom == 0.0:
            denom = 1e-12
        probs = masked_exp / denom
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
    def quantum_masked_self_attention(
        self,
        vec_Q: np.ndarray,
        vec_K: np.ndarray,
        vec_V: np.ndarray,
        mask_size: int = DEFAULT_REDUCE_DIM,
    ) -> np.ndarray:
        """Run the masked self-attention pipeline on Q/K/V vectors.

        Steps:
            1) Build a causal mask of length `mask_size` (flattened)
            2) Amplitude-encode Q, K, V
            3) Swap test between Q and K to get raw scores (bitstring dict)
            4) Apply masked softmax to the scores
            5) Weighted sum over duplicated V (placeholder for multiple states)
        """
        # 1) Causal mask (lower-triangular flattened)
        causal = make_causal_mask(mask_size)

        # 2) Encode
        qc_Q = self.amplitude_encode(vec_Q)
        qc_K = self.amplitude_encode(vec_K)
        qc_V = self.amplitude_encode(vec_V)

        # 3) Swap test -> raw scores
        raw_scores = self.measure_state(self.swap_test(qc_Q, qc_K))

        # 4) Masked softmax
        masked_probs = self.softmax_with_mask(raw_scores, causal)

        # 5) Weighted sum over V placeholders
        value_matrix = np.stack([vec_V, vec_V], axis=0)  # placeholder
        out = self.weighted_sum_of_values(masked_probs, value_matrix)
        return out
# endregion
