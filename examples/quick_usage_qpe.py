
"""
Quick Usage example for the QPE (Quantum Positional Encoding) module.
Run: python -m my_package.examples.quick_usage_qpe
"""
from __future__ import annotations
import numpy as np
import torch

from my_package.qpe import QuantumSinusoidalPositionalEncoding, QuantumLearnablePositionalEncoding

def main() -> None:
    # Sinusoidal QPE
    qpe_sin = QuantumSinusoidalPositionalEncoding(max_seq_len=32, d_model=8, base=10000)
    circuits = qpe_sin.encode_sequence([0, 1, 2, 3])
    print("Built", len(circuits), "circuits with", qpe_sin.n_qubits, "qubits each.")

    # Learnable QPE
    qpe_learn = QuantumLearnablePositionalEncoding(max_seq_len=16, n_qubits=3, n_layers=2)
    positions = [0, 5, 10]
    states = qpe_learn(positions)
    print("Learnable QPE states shape:", tuple(states.shape))  # (3, 2**3)

if __name__ == "__main__":
    main()
