
"""
Quick Usage for QViT patch embedding.
Run: python -m my_package.examples.quick_usage_qvit
"""
from __future__ import annotations
import numpy as np

from my_package.qvit import PatchPreprocessor, QViTPatchFeatureMap, create_ansatz

def main() -> None:
    # Fake "patches" for one image: 4 patches, each patch_dim=16 -> reduce to 2 features/patch
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(4, 16)).astype("float32")

    prep = PatchPreprocessor(n_components=2, feature_range=(-1, 1))
    patches_reduced = prep.transform_patches(patches)
    x_flat = prep.flatten_patches(patches_reduced)  # length = 4 * 2 = 8

    fmap_builder = QViTPatchFeatureMap(n_patches=4, n_qubits_per_patch=2)
    fmap, params = fmap_builder.build()
    bindings = fmap_builder.map_features_to_parameters(x_flat)

    # Build an ansatz compatible with the total qubits (CLS + patches)
    ansatz = create_ansatz(n_qubits=1 + 4 * 2, reps=2)

    # The resulting `fmap` and `ansatz` can be fed to a VQC or custom EstimatorQNN.
    print("Feature map qubits:", fmap.num_qubits, "Num params:", len(params))
    print("Ansatz qubits:", ansatz.num_qubits)

if __name__ == "__main__":
    main()
