
"""
Quick Usage for QSANN.
Run: python -m my_package.examples.quick_usage_qsann
"""
from __future__ import annotations
import torch
from my_package.qsann import QSALConfig, QSANN

def main() -> None:
    S, n, Denc, D = 3, 3, 1, 1
    cfg = QSALConfig(seq_len=S, n_qubits=n, enc_depth=Denc, var_depth=D)
    model = QSANN(num_layers=2, config=cfg)

    # Expected feature dimension per token = n*(Denc+2)
    d = n * (Denc + 2)
    x = torch.randn(2, S, d)  # (batch=2, seq=S, features=d)

    out = model(x)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(out.shape))

if __name__ == "__main__":
    main()
