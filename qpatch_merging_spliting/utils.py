
"""
Utilities and configuration for the Qpatch_merging_spliting module.
"""

from __future__ import annotations

# ---- Module-level configuration ("globals") ----
DEFAULT_ANSATZ_REPS: int = 2
DEFAULT_FM_REPS: int = 1

def require_positive(name: str, value: int) -> None:
    """Validate that an integer parameter is positive (> 0)."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")

def require_equal(name_a: str, val_a: int, name_b: str, val_b: int) -> None:
    """Validate equality between two integer expressions."""
    if val_a != val_b:
        raise ValueError(f"{name_a} ({val_a}) must equal {name_b} ({val_b}).")
