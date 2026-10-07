"""
Tile exported GeoTIFF scenes (RGB + thermal bands stacked) into paired
256x256 patches, normalize them, and split by SCENE (not by patch) into
train/val/test.

Expects each GeoTIFF in RAW_DIR to have 4 bands in order:
  [R, G, B, Thermal]  (matches RGB_BANDS + [THERMAL_BAND] order in gee_export.py)

Output: .npy files under TILES_DIR/{split}/{scene}_{row}_{col}.npy, each
containing a (4, PATCH_SIZE, PATCH_SIZE) float32 array: channels 0-2 are
normalized RGB in [0, 1], channel 3 is normalized thermal in [0, 1].
"""

import os
import sys
import json
import glob
import random
import numpy as np
import rasterio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DIR, TILES_DIR, SPLIT_FILE, PATCH_SIZE, PATCH_STRIDE,
    TRAIN_SCENES_FRAC, VAL_SCENES_FRAC,
)

# Reasonable clipping ranges for normalization -- inspect your actual
# exported scenes (print min/max first) and adjust if results look off.
RGB_REFLECTANCE_RANGE = (0.0, 0.3)
THERMAL_KELVIN_RANGE = (260.0, 330.0)  # ~ -13C to 57C, covers most land surfaces


def normalize(arr, value_range):
    lo, hi = value_range
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


def tile_scene(path, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    """Yield (row, col, patch) for a single scene GeoTIFF."""
    with rasterio.open(path) as src:
        data = src.read()  # shape (4, H, W): R, G, B, Thermal

    r, g, b, thermal = data[0], data[1], data[2], data[3]
    rgb = np.stack([r, g, b], axis=0)

    rgb_norm = normalize(rgb, RGB_REFLECTANCE_RANGE)
    thermal_norm = normalize(thermal[None, ...], THERMAL_KELVIN_RANGE)

    stacked = np.concatenate([rgb_norm, thermal_norm], axis=0).astype(np.float32)
    _, h, w = stacked.shape

    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            patch = stacked[:, row:row + patch_size, col:col + patch_size]
            # skip mostly-empty/no-data patches (reprojection corners etc.)
            if (patch[:3] <= 0).mean() > 0.3:
                continue
            yield row, col, patch


def build_splits(scene_names, seed=42):
    rng = random.Random(seed)
    scenes = list(scene_names)
    rng.shuffle(scenes)

    n = len(scenes)
    n_train = max(1, int(n * TRAIN_SCENES_FRAC))
    n_val = max(1, int(n * VAL_SCENES_FRAC))

    return {
        "train": scenes[:n_train],
        "val": scenes[n_train:n_train + n_val],
        "test": scenes[n_train + n_val:],
    }


def main():
    scene_paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.tif")))
    if not scene_paths:
        raise FileNotFoundError(
            f"No .tif files found in {RAW_DIR}/. Download your GEE exports "
            f"from Drive into this folder first."
        )

    scene_names = [os.path.splitext(os.path.basename(p))[0] for p in scene_paths]
    splits = build_splits(scene_names)

    with open(SPLIT_FILE, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Scene split written to {SPLIT_FILE}: "
          f"{len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test scenes")

    scene_to_split = {s: split for split, names in splits.items() for s in names}

    total_tiles = 0
    for path, name in zip(scene_paths, scene_names):
        split = scene_to_split[name]
        out_dir = os.path.join(TILES_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        count = 0
        for row, col, patch in tile_scene(path):
            out_path = os.path.join(out_dir, f"{name}_{row}_{col}.npy")
            np.save(out_path, patch)
            count += 1
        total_tiles += count
        print(f"  {name} ({split}): {count} patches")

    print(f"\nDone. {total_tiles} total patches written to {TILES_DIR}/")


if __name__ == "__main__":
    main()
