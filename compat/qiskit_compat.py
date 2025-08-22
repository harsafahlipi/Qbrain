
from __future__ import annotations
HAS_AER = False
HAS_PRIMITIVES = False
HAS_QML = False
try:
    from qiskit_aer import AerSimulator  # type: ignore
    HAS_AER = True
except Exception:
    try:
        from qiskit.providers.aer import AerSimulator  # type: ignore
        HAS_AER = True
    except Exception:
        AerSimulator = None  # type: ignore

Estimator = None  # type: ignore
Sampler = None  # type: ignore
try:
    from qiskit.primitives import Estimator as _E, Sampler as _S  # type: ignore
    Estimator, Sampler = _E, _S
    HAS_PRIMITIVES = True
except Exception:
    try:
        from qiskit_aer.primitives import Estimator as _E, Sampler as _S  # type: ignore
        Estimator, Sampler = _E, _S
        HAS_PRIMITIVES = True
    except Exception:
        pass
try:
    import qiskit_machine_learning as _qml  # noqa: F401
    HAS_QML = True
except Exception:
    pass
__all__ = ["HAS_AER", "HAS_PRIMITIVES", "HAS_QML", "AerSimulator", "Estimator", "Sampler"]
