
from __future__ import annotations
HAS_PENNYLANE = False
try:
    import pennylane as qml  # noqa: F401
    HAS_PENNYLANE = True
except Exception:
    qml = None  # type: ignore
__all__ = ["HAS_PENNYLANE", "qml"]
