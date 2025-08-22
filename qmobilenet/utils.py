
"""
Utilities and configuration for the QMobileNet module.
"""

from __future__ import annotations

# ---- Module-level configuration ("globals") ----
DEFAULT_USE_QUANTUM: bool = True
DEFAULT_QUANTUM_RATIO: float = 0.25

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))
