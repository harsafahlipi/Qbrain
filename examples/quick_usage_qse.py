
"""
Quick Usage example for the QSE module.
Run: python -m my_package.examples.quick_usage_qse
"""
from __future__ import annotations
import numpy as np

from my_package.qse import QuantumSEBlock, get_estimator

def main() -> None:
    n_channels = 4
    se = QuantumSEBlock(n_channels=n_channels, reduction_ratio=2)

    # Random demo inputs (angles) and params
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2*np.pi, size=n_channels).tolist()
    n_params = len(se.fc1_params) + len(se.fc2_params)
    params = rng.uniform(-np.pi, np.pi, size=n_params).tolist()

    est = get_estimator()
    circ, weights = se.forward(angles, params, estimator=est)
    print("Weights shape:", weights.shape, "Circuit qubits:", circ.num_qubits)

if __name__ == "__main__":
    main()
