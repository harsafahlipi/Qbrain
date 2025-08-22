
"""
Quick Usage for qcombined_encoding.
Run: python -m my_package.examples.quick_usage_qcombined_encoding

Note: This example intentionally avoids printing. Inspect returned metrics in a debugger
or modify to log as desired.
"""
from __future__ import annotations
import torch
from my_package.qcombined_encoding import CombinedEncodingConfig, QuantumClassifier, train_one_epoch, evaluate

def main() -> dict:
    cfg = CombinedEncodingConfig(n_rot=2, n_amp=2, n_layers=2)
    model = QuantumClassifier(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Dummy data (batch=8): rotation features (n_rot), amplitude features (<= 2**n_amp), binary labels
    xb_rot = torch.randn(8, cfg.n_rot)
    xb_amp = torch.randn(8, 2)  # shorter than 2**n_amp -> will be padded inside
    yb = (torch.rand(8) > 0.5).float()

    train_metrics = train_one_epoch(model, opt, loss_fn, xb_rot, xb_amp, yb)
    eval_metrics = evaluate(model, loss_fn, xb_rot, xb_amp, yb)

    # Return metrics for external inspection
    return {"train": train_metrics, "eval": eval_metrics}

if __name__ == "__main__":
    main()
