
"""
Quick Usage example for the QLayerNormalization module.
Run: python -m my_package.examples.quick_usage_qlayer_normalization
"""
from __future__ import annotations
import numpy as np

from my_package.qlayer_normalization import QuantumLayerNormalization, VariationalQuantumLayerNorm

def main() -> None:
    n_qubits, n_layers = 3, 2
    qln = QuantumLayerNormalization(n_qubits=n_qubits, n_layers=n_layers)

    # Inputs and params
    x = np.random.uniform(0, 2*np.pi, size=n_qubits)
    gamma = np.ones(n_qubits * n_layers, dtype=np.float64)
    beta = np.zeros(n_qubits * n_layers, dtype=np.float64)

    z_exp = qln.normalize_quantum_state(x, gamma, beta)
    print("Z expectations:", z_exp, "shape:", z_exp.shape)

    # Variational form on a probability vector
    vln = VariationalQuantumLayerNorm(n_qubits=3, learning_rate=1e-2)
    probs = np.random.dirichlet(np.ones(2**3))
    out = vln.forward(probs)
    print("Variational output shape:", out.shape)

if __name__ == "__main__":
    main()
