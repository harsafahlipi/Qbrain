
"""
Quick Usage for QTransformer Decoder.
Run: python -m my_package.examples.quick_usage_qtransformer_decoder
"""
from __future__ import annotations
import numpy as np
import torch

from my_package.qtransformer_decoder import create_qnn, HybridModel

def main() -> None:
    n_qubits, n_layers = 4, 2
    # Build QNN (requires qiskit-machine-learning + Estimator)
    qnn = create_qnn(n_qubits=n_qubits, n_layers=n_layers)
    model = HybridModel(qnn, num_classes=5)
    # Dummy batch (B, n_qubits) of angles
    x = torch.tensor(np.random.uniform(-1, 1, size=(2, n_qubits)).astype("float32"))
    y = model(x)
    print("Output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
