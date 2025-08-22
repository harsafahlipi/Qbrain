
"""
Utilities and configuration for the QGhost (Quantum Ghost Module Block) module.
"""

from __future__ import annotations

# ---- Module-level configuration ("globals") ----
DEFAULT_FEATURE_MAP_REPS: int = 1
DEFAULT_ANSATZ_REPS: int = 2

def require_positive(name: str, value: int) -> None:
    """Validate that an integer parameter is positive (> 0)."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
