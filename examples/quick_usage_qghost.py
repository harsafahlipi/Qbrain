
"""
Quick Usage example for the QGhost module.
Run: python -m my_package.examples.quick_usage_qghost
"""
from __future__ import annotations

from my_package.qghost import QuantumGhostBlock, create_quantum_ghost_block

def main() -> None:
    # Using the convenience function (ghost mode)
    fm, ans, qc = create_quantum_ghost_block(num_qubits=4, feature_map_reps=1, ansatz_reps=2, mode="ghost")
    print("Ghost block qubits:", qc.num_qubits)

    # Using the class (real_amplitudes mode)
    block = QuantumGhostBlock(num_qubits=3, feature_map_reps=1, ansatz_reps=1, mode="real_amplitudes")
    fm2, ans2, qc2 = block.build()
    print("RealAmplitudes block qubits:", qc2.num_qubits)

if __name__ == "__main__":
    main()
