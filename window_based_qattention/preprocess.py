
"""
Minimal image → feature-vector extractor for window-based quantum attention.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def image_to_feature_vector(image_path: str, size: Tuple[int, int] = (4, 4)) -> np.ndarray:
    """
    Load a grayscale image, resize, flatten, and return as float32 vector (no normalization).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image at path: {image_path}")
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized.flatten().astype(np.float32, copy=False)
