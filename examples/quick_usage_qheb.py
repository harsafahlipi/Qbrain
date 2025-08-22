
"""
Quick Usage example for the QHEB (Quantum Hierarchical Embedding Block) module.
Run: python -m my_package.examples.quick_usage_qheb
"""
from __future__ import annotations

from my_package.qheb import QuantumHierarchicalEmbeddingBlock, create_qheb_feature_map

def main() -> None:
    # Class API (returns params and circuit)
    params, fmap = QuantumHierarchicalEmbeddingBlock(num_qubits=4, reps=2).build()
    print("Parameters:", list(params))
    print("Qubits:", fmap.num_qubits)

    # Convenience function (returns just the circuit)
    fmap2 = create_qheb_feature_map(num_qubits=3, reps=1)
    print("Qubits (convenience):", fmap2.num_qubits)

if __name__ == "__main__":
    main()
