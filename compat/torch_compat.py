
from __future__ import annotations
HAS_TORCH = False
try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
__all__ = ["HAS_TORCH", "torch"]
