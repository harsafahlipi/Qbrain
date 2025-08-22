
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class QResNetConfig:
    n_qubits: int = 4
    n_quantum_layers: int = 2
    num_classes: int = 10
    use_quantum_eval: bool = False
    shots: int = 1000
