
"""
QCBAM (Quantum Channel & Spatial Attention) — Core Module
========================================================

This module implements a quantum-inspired QCBAM pipeline using Qiskit.
- Docstrings (Google style), DRY, SOLID, Type hints, KISS.
- Strategy Pattern for feature extraction.
- Global configuration via utils.py constants.
"""

from __future__ import annotations

# region ========== Standard Library & Typing ==========
from typing import Dict, Iterable, Protocol, Tuple, runtime_checkable, Optional
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
    """Simple (fast) feature extractor: resize -> flatten -> slice -> normalize.

    Adheres to KISS and can be replaced by more advanced strategies like PCA or CNN.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (4, 4),
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


# region ========== Core QCBAM Class ==========
class QCBAM:
    """Quantum QCBAM (Channel & Spatial Attention) using Qiskit.

    SOLID:
        - Single Responsibility: each method has a single concern.
        - Open/Closed: pluggable FeatureExtractor; backend injection.
        - Liskov Substitution: any FeatureExtractor that satisfies the Protocol works.
        - Interface Segregation: only one required method on the strategy.
        - Dependency Inversion: depends on abstraction (FeatureExtractor).

    Design:
        - Strategy Pattern for feature extraction.
        - Backend (AerSimulator) injected for testability and extension.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        shots: int = DEFAULT_SHOTS,
        backend: Optional[AerSimulator] = None,
        sim_method: str = DEFAULT_SIM_METHOD,
        max_threads: int = DEFAULT_MAX_THREADS,
    ) -> None:
        """Initialize QCBAM with configurable dependencies."""
        require_positive("shots", shots)
        self.feature_extractor = feature_extractor or SimpleResizeExtractor()
        self.shots = shots
        self.backend = backend or AerSimulator(method=sim_method, max_parallel_threads=max_threads)

    # ---- Feature Extraction Facade ----
    def image_to_feature_vector(self, image_path: str) -> np.ndarray:
        """Facade forwarding to feature_extractor strategy."""
        return self.feature_extractor.image_to_feature_vector(image_path)

    # ---- Quantum Encoding ----
    @staticmethod
    def amplitude_encode(vector: np.ndarray) -> QuantumCircuit:
        """Amplitude-encode a real vector into a quantum state (with padding & normalization)."""
        vec = np.asarray(vector, dtype=np.float64).ravel()
        vec, n_qubits = pad_to_pow2(vec)
        if np.allclose(vec, 0):
            raise ValueError("Cannot amplitude-encode a zero vector.")
        vec = normalize_vector(vec)
        qc = QuantumCircuit(n_qubits)
        qc.initialize(vec, range(n_qubits))
        return qc

    # ---- Channel Attention ----
    @staticmethod
    def quantum_inner_product(qc_q: QuantumCircuit, qc_k: QuantumCircuit) -> QuantumCircuit:
        """Entangle |Q> and |K> via CNOTs and measure the second register."""
        if qc_q.num_qubits != qc_k.num_qubits:
            raise ValueError("Circuits must have the same number of qubits.")
        n = qc_q.num_qubits
        qc = QuantumCircuit(2 * n, n)
        qc.compose(qc_q, qubits=range(n), inplace=True)
        qc.compose(qc_k, qubits=range(n, 2 * n), inplace=True)
        for i in range(n):
            qc.cx(i, n + i)
        for i in range(n):
            qc.measure(n + i, i)
        return qc

    # ---- Spatial Attention (QFT-like) ----
    @staticmethod
    def spatial_attention(qc_in: QuantumCircuit) -> QuantumCircuit:
        """Apply a simple QFT-like transform and measure all qubits."""
        n = qc_in.num_qubits
        qc_qft = QuantumCircuit(n, n)
        qc_qft.compose(qc_in, inplace=True)
        for i in range(n):
            qc_qft.h(i)
            for j in range(i + 1, n):
                qc_qft.cp(np.pi / (2 ** (j - i)), j, i)
        for i in range(n // 2):
            qc_qft.swap(i, n - 1 - i)
        for i in range(n):
            qc_qft.measure(i, i)
        return qc_qft

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

    # ---- Full Pipeline ----
    def quantum_CBAM(self, vec_Q: np.ndarray, vec_K: np.ndarray, vec_V: np.ndarray) -> np.ndarray:
        """Run the full QCBAM attention pipeline on Q/K/V vectors."""
        qc_Q = self.amplitude_encode(vec_Q)
        qc_K = self.amplitude_encode(vec_K)
        qc_V = self.amplitude_encode(vec_V)

        ch_attn = self.softmax(self.measure_state(self.quantum_inner_product(qc_Q, qc_K)))
        sp_attn = self.softmax(self.measure_state(self.spatial_attention(qc_V)))

        states = set(ch_attn) | set(sp_attn)
        combined = {s: ch_attn.get(s, 0.0) * sp_attn.get(s, 0.0) for s in states}

        value_vectors = np.stack([vec_V, vec_V], axis=0)  # placeholder example
        out = np.zeros_like(vec_V, dtype=np.float64)
        for bitstr, weight in combined.items():
            try:
                idx = int(bitstr, 2) % len(value_vectors)
                out += weight * value_vectors[idx]
            except ValueError:
                continue
        return out.astype(np.float32)
# endregion
