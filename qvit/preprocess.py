
"""
Preprocessing utilities for QViT patch embedding.

This module provides per-sample PCA reduction to a fixed number of features per patch
and scaling to a bounded range, mirroring the original snippet's intent but in a
clean, reusable form.
"""

from __future__ import annotations

# region ========== Typing / Stdlib ==========
from typing import Tuple
# endregion

# region ========== Third-Party Imports ==========
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# endregion


class PatchPreprocessor:
    """
    Per-sample patch preprocessor: PCA → MinMax scaling → flatten.
    KISS: we fit PCA and scaler *per sample* (matches the provided snippet).
    """

    def __init__(self, n_components: int = 2, feature_range: Tuple[float, float] = (-1.0, 1.0)) -> None:
        self.n_components = int(n_components)
        self.feature_range = feature_range

    # region ---- API ----
    def transform_patches(self, patches: np.ndarray) -> np.ndarray:
        """
        Reduce each sample's patches to n_components and scale to feature_range.
        Args:
            patches: array of shape (n_patches, patch_dim)
        Returns:
            reduced_and_scaled: array of shape (n_patches, n_components)
        """
        if patches.ndim != 2:
            raise ValueError("patches must be 2D: (n_patches, patch_dim)")
        pca = PCA(n_components=self.n_components)
        reduced = pca.fit_transform(patches)  # fit per sample
        scaler = MinMaxScaler(feature_range=self.feature_range)
        scaled = scaler.fit_transform(reduced)
        return scaled

    def flatten_patches(self, patches_reduced: np.ndarray) -> np.ndarray:
        """
        Flatten reduced patches into a single feature vector.
        Args:
            patches_reduced: (n_patches, n_components)
        Returns:
            flat: (n_patches * n_components,)
        """
        if patches_reduced.ndim != 2:
            raise ValueError("patches_reduced must be 2D: (n_patches, n_components)")
        return patches_reduced.reshape(-1).astype(np.float32, copy=False)
    # endregion
