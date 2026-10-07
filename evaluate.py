"""
Compute PSNR, SSIM, and LPIPS between generated RGB and ground-truth RGB
on the held-out test split.

Usage:
    python evaluate.py --checkpoint checkpoints/se_resunet_full.pt
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

from config import TILES_DIR
from data.dataset import IRRGBDataset
from models.se_resunet import SEResUNet
from utils import load_checkpoint


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tiles_dir", default=TILES_DIR)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = IRRGBDataset(args.tiles_dir, "test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SEResUNet().to(device)
    load_checkpoint(model, None, args.checkpoint, device=device)
    model.eval()

    lpips_fn = lpips.LPIPS(net="alex").to(device)

    psnr_scores, ssim_scores, lpips_scores = [], [], []

    with torch.no_grad():
        for thermal, rgb in test_loader:
            thermal, rgb = thermal.to(device), rgb.to(device)
            pred = model(thermal)

            lp = lpips_fn(pred * 2 - 1, rgb * 2 - 1)  # LPIPS expects [-1, 1]
            lpips_scores.extend(lp.squeeze().cpu().numpy().reshape(-1).tolist())

            pred_np = pred.cpu().numpy()
            rgb_np = rgb.cpu().numpy()
            for p_img, r_img in zip(pred_np, rgb_np):
                p_img = np.transpose(p_img, (1, 2, 0))
                r_img = np.transpose(r_img, (1, 2, 0))
                psnr_scores.append(psnr(r_img, p_img, data_range=1.0))
                ssim_scores.append(ssim(r_img, p_img, data_range=1.0, channel_axis=2))

    print(f"Test set: {len(psnr_scores)} patches")
    print(f"PSNR:  {np.mean(psnr_scores):.3f} dB")
    print(f"SSIM:  {np.mean(ssim_scores):.4f}")
    print(f"LPIPS: {np.mean(lpips_scores):.4f} (lower is better)")


if __name__ == "__main__":
    main()
