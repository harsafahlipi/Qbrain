
"""
qcombined_encoding — Aggregator Module
=====================================

Exports:
- CombinedEncodingConfig
- QuantumClassifier
- train_one_epoch, evaluate
"""

from __future__ import annotations
from .config import CombinedEncodingConfig
from .model import QuantumClassifier
from .training import train_one_epoch, evaluate

__all__ = ["CombinedEncodingConfig", "QuantumClassifier", "train_one_epoch", "evaluate"]
