"""
Training loop for SE-ResUNet IR2RGB, with mixed precision and periodic
checkpointing (designed around Colab/Kaggle session limits -- point
--checkpoint_dir at a mounted Google Drive path so a disconnect doesn't
lose progress).

Usage:
    python train.py --checkpoint_dir /content/drive/MyDrive/ir2rgb_checkpoints

Ablation runs (matches the project's ablation table):
    python train.py --ablation baseline     # L1 only
    python train.py --ablation perceptual   # L1 + perceptual
    python train.py --ablation full         # L1 + perceptual + structural
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import (
    TILES_DIR, BATCH_SIZE, NUM_EPOCHS, LR, CHECKPOINT_DIR,
    CHECKPOINT_EVERY_STEPS, LAMBDA_L1, LAMBDA_PERCEPTUAL, LAMBDA_STRUCTURAL,
)
from data.dataset import IRRGBDataset
from models.se_resunet import SEResUNet
from models.losses import CombinedLoss
from utils import save_checkpoint, load_checkpoint, count_parameters


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles_dir", default=TILES_DIR)
    p.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    p.add_argument("--ablation", default="full", choices=["baseline", "perceptual", "full"],
                    help="which loss terms to enable, for the ablation table")
    return p.parse_args()


def build_loss(ablation):
    weights = {
        "baseline":   dict(lambda_l1=1.0, lambda_perceptual=0.0, lambda_structural=0.0),
        "perceptual": dict(lambda_l1=1.0, lambda_perceptual=LAMBDA_PERCEPTUAL, lambda_structural=0.0),
        "full":       dict(lambda_l1=LAMBDA_L1, lambda_perceptual=LAMBDA_PERCEPTUAL, lambda_structural=LAMBDA_STRUCTURAL),
    }[ablation]
    return CombinedLoss(**weights)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = IRRGBDataset(args.tiles_dir, "train", augment=True)
    val_ds = IRRGBDataset(args.tiles_dir, "val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SEResUNet().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = build_loss(args.ablation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device == "cuda"))

    start_epoch, step = 0, 0
    if args.resume:
        start_epoch, step = load_checkpoint(model, optimizer, args.resume, device=device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, step {step}")

    ckpt_path = os.path.join(args.checkpoint_dir, f"se_resunet_{args.ablation}.pt")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {"l1": 0.0, "perceptual": 0.0, "structural": 0.0, "total": 0.0}

        for thermal, rgb in train_loader:
            thermal, rgb = thermal.to(device), rgb.to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device == "cuda")):
                pred = model(thermal)
                loss, parts = criterion(pred, rgb, thermal)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k in running:
                running[k] += parts[k]
            step += 1

            if step % CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(model, optimizer, epoch, step, ckpt_path)
                print(f"  [checkpoint] epoch {epoch} step {step} -> {ckpt_path}")

        n = len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"L1 {running['l1']/n:.4f} | Perc {running['perceptual']/n:.4f} | "
              f"Struct {running['structural']/n:.4f} | Total {running['total']/n:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for thermal, rgb in val_loader:
                thermal, rgb = thermal.to(device), rgb.to(device)
                pred = model(thermal)
                loss, _ = criterion(pred, rgb, thermal)
                val_loss += loss.item()
        print(f"  Val loss: {val_loss/len(val_loader):.4f}")

        save_checkpoint(model, optimizer, epoch + 1, step, ckpt_path)

    print(f"Training complete. Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
