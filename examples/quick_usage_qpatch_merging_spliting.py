
"""
Quick Usage example for the Qpatch_merging_spliting module.
Run: python -m my_package.examples.quick_usage_qpatch_merging_spliting
"""
from __future__ import annotations

from my_package.qpatch_merging_spliting import QuantumPatchMergingSplitingBlock, patch_merging_block

def main() -> None:
    # Using the convenience function (original-style API)
    fmap, ans, xin, win, full = patch_merging_block(num_qubits=8, patch_size=2, num_patches=4, reps=2)
    print("Feature-map qubits:", fmap.num_qubits, "| Ansatz params:", len(win))

    # Using the class
    block = QuantumPatchMergingSplitingBlock(num_qubits=6, patch_size=3, num_patches=2, ansatz_reps=1)
    fmap2, ans2, xin2, win2, full2 = block.build()
    print("Full circuit qubits:", full2.num_qubits)

if __name__ == "__main__":
    main()
