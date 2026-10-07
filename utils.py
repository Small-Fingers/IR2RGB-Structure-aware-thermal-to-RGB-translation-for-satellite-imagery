"""Checkpointing and small helpers shared across train.py / evaluate.py."""

import os
import torch


def save_checkpoint(model, optimizer, epoch, step, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
