
"""
Minimal image → feature-vector extractor for Slot Attention examples.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2

from .utils import DEFAULT_IMAGE_SIZE, DEFAULT_REDUCE_DIM


def image_to_feature_vector(image_path: str, size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, reduce_dim: int = DEFAULT_REDUCE_DIM) -> np.ndarray:
    """
    Load a grayscale image, resize, flatten, truncate to reduce_dim, and L2-normalize.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image at path: {image_path}")
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    flat = resized.flatten().astype(np.float32)
    if flat.size > reduce_dim:
        flat = flat[:reduce_dim]
    norm = float(np.linalg.norm(flat))
    return (flat / norm) if norm != 0.0 else flat
