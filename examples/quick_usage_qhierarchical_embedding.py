
"""
Quick Usage for Qhierarchical_embedding.
Run: python -m my_package.examples.quick_usage_qhierarchical_embedding

Notes:
- This example returns objects for inspection and avoids prints by design.
"""
from __future__ import annotations

from qiskit.circuit.library import RealAmplitudes
from my_package.qhierarchical_embedding import create_qheb_feature_map

def main():
    num_qubits = 4
    feature_map = create_qheb_feature_map(num_qubits=num_qubits, reps=2)
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3)
    # Return for inspection
    return {"feature_map": feature_map, "ansatz": ansatz}

if __name__ == "__main__":
    main()
