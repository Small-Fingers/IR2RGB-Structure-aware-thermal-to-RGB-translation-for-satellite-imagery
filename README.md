# IR2RGB: Structure-aware thermal-to-RGB translation for satellite imagery

Converts grayscale Landsat 8/9 thermal imagery into realistic color RGB,
using an SE-ResUNet trained with a combined L1 + perceptual + structural
consistency loss. Evaluated on both image quality (PSNR/SSIM/LPIPS) and
downstream object detection (YOLOv8).

## Project structure

```
ir2rgb/
  config.py             # all paths, hyperparameters, band names
  data/
    gee_export.py       # pull Landsat scenes via Google Earth Engine
    tiling.py            # tile scenes into 256x256 patches, split by scene
    dataset.py            # PyTorch Dataset
  models/
    se_resunet.py          # SE-ResUNet architecture
    losses.py                # L1 + perceptual + structural consistency loss
  train.py                    # training loop (mixed precision, checkpointing)
  evaluate.py                  # PSNR / SSIM / LPIPS on test set
  detect_eval.py                 # YOLOv8 downstream detection comparison
  utils.py                        # checkpoint save/load helpers
```

## Setup (Google Colab)

```bash
!pip install -q rasterio scikit-image lpips ultralytics earthengine-api
```

Mount Drive so checkpoints survive a disconnect:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 1: Pull data from Earth Engine

```bash
python data/gee_export.py
```

Authenticates with Earth Engine, finds the least-cloudy Landsat 8/9 scene
for each location in `config.SCENE_CENTERS`, and exports a 4-band (R, G,
B, thermal) GeoTIFF per scene to your Google Drive under `ir2rgb_raw/`.
Download these into a local `data_raw/` folder (or mount Drive directly
and point `RAW_DIR` at it).

## Step 2: Tile into patches

```bash
python data/tiling.py
```

Tiles each scene into 256x256 patches, normalizes RGB (reflectance) and
thermal (Kelvin) into [0, 1], and writes `splits.json` assigning whole
scenes to train/val/test — never splitting one scene's patches across
sets, to avoid data leakage.

## Step 3: Train

```bash
python train.py --checkpoint_dir /content/drive/MyDrive/ir2rgb_checkpoints
```

Reproduce the project's ablation table with `--ablation`:

```bash
python train.py --ablation baseline     # L1 only
python train.py --ablation perceptual   # L1 + perceptual
python train.py --ablation full         # L1 + perceptual + structural (full model)
```

SE blocks are always present in the architecture, to keep the codebase
simple. If you want a true "no SE" ablation row for comparison, swap
`SEBlock` for `nn.Identity()` in `models/se_resunet.py` for that one run.

Checkpoints save every `CHECKPOINT_EVERY_STEPS` steps, not just at epoch
end, since Colab disconnects don't wait for epoch boundaries.

## Step 4: Evaluate image quality

```bash
python evaluate.py --checkpoint /path/to/se_resunet_full.pt
```

Reports PSNR, SSIM, and LPIPS on the held-out test scenes.

## Step 5: Downstream object detection evaluation

```bash
python detect_eval.py --checkpoint /path/to/se_resunet_full.pt
```

Runs YOLOv8 on generated vs. real RGB test patches and compares detection
counts. **Read before you run this:** stock YOLOv8 weights are
COCO-trained and have no "building" class. Two honest options, both
explained in the script's own output:

1. Fine-tune YOLOv8 on a labeled building dataset (e.g. SpaceNet) for a
   true building-detection comparison, or
2. Report detections on COCO classes actually present in your scenes
   (car, truck, boat) as a proxy for "does the generated image preserve
   detectable structure," and say so explicitly in your report.

Either is defensible — just don't silently claim "building detection" if
you ran stock weights without checking what classes they output.

## Notes on scope and honesty for your report

- **"Physics-constrained" became "structural consistency."** The
  structural loss enforces edge/boundary alignment between thermal input
  and RGB output. It does not claim physics uniquely determines color —
  it doesn't (two materials at the same temperature can be any color).
- **No GAN.** This codebase uses a perceptual loss instead of an
  adversarial loss, to avoid GAN training instability within an 8-10 week
  timeline. Noted as explicit future work.
- **Single output head.** One decoder produces RGB directly, with the
  structural loss applied to that same output, rather than a separate SR
  branch — simpler to train, same conceptual benefit.

## What's CV vs. image processing in this codebase

- Image processing: `gee_export.py` (atmospheric correction via Landsat's
  own scale factors), `tiling.py` (normalization, patch extraction).
- Computer vision: `se_resunet.py` + `losses.py` + `train.py` (the learned
  thermal-to-RGB mapping and its structural/perceptual constraints),
  `detect_eval.py` (object detection on the learned output).
