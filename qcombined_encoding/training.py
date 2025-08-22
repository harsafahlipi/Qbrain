
"""
Training utilities for qcombined_encoding — no prints, returns metrics.
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer


def train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    xb_rot: torch.Tensor,
    xb_amp: torch.Tensor,
    yb: torch.Tensor,
) -> Dict[str, float]:
    """
    Train on a single epoch worth of tensor batches (already prepared tensors).
    No dataloaders here to keep things simple and framework-agnostic.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model.forward_batch(xb_rot, xb_amp)  # (batch,)
    loss = loss_fn(logits, yb.float())
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == yb).float().mean().item()

    return {"loss": float(loss.item()), "accuracy": acc}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loss_fn: nn.Module,
    xb_rot: torch.Tensor,
    xb_amp: torch.Tensor,
    yb: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate model — returns metrics dict with loss and accuracy.
    """
    model.eval()
    logits = model.forward_batch(xb_rot, xb_amp)
    loss = loss_fn(logits, yb.float())
    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == yb).float().mean().item()
    return {"loss": float(loss.item()), "accuracy": acc}
