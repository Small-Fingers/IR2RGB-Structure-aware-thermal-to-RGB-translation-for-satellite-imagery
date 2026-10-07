"""
Downstream evaluation: run a pretrained YOLOv8 detector on generated RGB
vs. real RGB test patches and compare detection performance. This is what
proves the project's practical claim -- that generated imagery is actually
useful for object detection, not just visually plausible.

Usage:
    python detect_eval.py --checkpoint checkpoints/se_resunet_full.pt

Important: see the printed note at the end about stock YOLOv8's class set
before you report these numbers as "building detection" in your writeup.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import TILES_DIR
from data.dataset import IRRGBDataset
from models.se_resunet import SEResUNet
from utils import load_checkpoint


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tiles_dir", default=TILES_DIR)
    p.add_argument("--out_dir", default="detect_eval_images")
    p.add_argument("--yolo_weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    return p.parse_args()


def tensor_to_pil(img_tensor):
    arr = (img_tensor.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_dir = os.path.join(args.out_dir, "generated")
    real_dir = os.path.join(args.out_dir, "real")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    test_ds = IRRGBDataset(args.tiles_dir, "test", augment=False)

    model = SEResUNet().to(device)
    load_checkpoint(model, None, args.checkpoint, device=device)
    model.eval()

    with torch.no_grad():
        for i in range(len(test_ds)):
            thermal, rgb = test_ds[i]
            pred = model(thermal.unsqueeze(0).to(device)).squeeze(0).cpu()
            tensor_to_pil(pred).save(os.path.join(gen_dir, f"{i:04d}.png"))
            tensor_to_pil(rgb).save(os.path.join(real_dir, f"{i:04d}.png"))

    detector = YOLO(args.yolo_weights)
    results_gen = detector.predict(source=gen_dir, conf=args.conf, verbose=False)
    results_real = detector.predict(source=real_dir, conf=args.conf, verbose=False)

    def count_detections(results):
        return sum(len(r.boxes) for r in results)

    n_gen = count_detections(results_gen)
    n_real = count_detections(results_real)

    print(f"Total detections on generated RGB: {n_gen}")
    print(f"Total detections on real RGB:      {n_real}")
    print(f"Retention ratio (generated/real):  {n_gen / max(n_real, 1):.3f}")
    print(
        "\nNote: stock YOLOv8 weights are COCO-trained and don't include a "
        "'building' class. For a true building-detection comparison, either "
        "(a) fine-tune YOLOv8 on a labeled building dataset (e.g. SpaceNet), "
        "or (b) report detections on COCO classes actually present in your "
        "scenes (car, truck, boat) as a proxy for 'does the generated image "
        "preserve detectable structure', and say so explicitly in your report."
    )


if __name__ == "__main__":
    main()
