"""PyTorch Dataset for paired thermal -> RGB patches produced by tiling.py."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class IRRGBDataset(Dataset):
    """
    Loads (4, H, W) .npy patches where channels 0-2 are normalized RGB
    ground truth and channel 3 is normalized thermal input.

    Returns (thermal, rgb) as separate tensors:
        thermal: (1, H, W) float32 in [0, 1]
        rgb:     (3, H, W) float32 in [0, 1]
    """

    def __init__(self, tiles_dir, split, augment=False):
        self.paths = sorted(glob.glob(os.path.join(tiles_dir, split, "*.npy")))
        if not self.paths:
            raise FileNotFoundError(
                f"No patches found in {tiles_dir}/{split}/. Run data/tiling.py first."
            )
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        patch = np.load(self.paths[idx])  # (4, H, W)
        rgb = patch[:3]
        thermal = patch[3:4]

        if self.augment:
            if np.random.rand() < 0.5:
                rgb = rgb[:, :, ::-1].copy()
                thermal = thermal[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                rgb = rgb[:, ::-1, :].copy()
                thermal = thermal[:, ::-1, :].copy()

        return torch.from_numpy(thermal), torch.from_numpy(rgb)
