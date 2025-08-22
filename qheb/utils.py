
"""
Utilities and configuration for the QHEB (Quantum Hierarchical Embedding Block) module.
"""

from __future__ import annotations

# ---- Module-level configuration ("globals") ----
DEFAULT_REPS: int = 2

def require_positive(name: str, value: int) -> None:
    """Validate that an integer parameter is positive (> 0)."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
