"""
Combined loss for IR2RGB training:
  1. L1 reconstruction loss
  2. Perceptual loss (VGG16 features) for realism
  3. Structural consistency loss (Sobel-edge correlation between thermal
     input and generated RGB luminance) to keep boundaries geometrically
     faithful -- this is the "structure-aware" term, deliberately scoped
     as structural/geometric consistency rather than a claim that physics
     uniquely determines color (it doesn't).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG16-feature L1 distance between generated and target RGB."""

    def __init__(self, layers=("3", "8", "15")):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layers = set(layers)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def _extract(self, x):
        x = self._normalize(x)
        feats = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                feats.append(x)
        return feats

    def forward(self, pred, target):
        pred_feats = self._extract(pred)
        target_feats = self._extract(target)
        loss = 0.0
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pf, tf)
        return loss / len(pred_feats)


class StructuralConsistencyLoss(nn.Module):
    """
    Penalizes mismatch between Sobel edges of the thermal input and Sobel
    edges of the generated RGB's luminance channel. Keeps boundaries
    (roads, buildings, coastlines) in the right place, not just the right
    average color.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("kx", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("ky", sobel_y.view(1, 1, 3, 3))

    def _edges(self, gray):
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def _luminance(self, rgb):
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def forward(self, pred_rgb, thermal):
        pred_edges = self._edges(self._luminance(pred_rgb))
        thermal_edges = self._edges(thermal)
        return F.l1_loss(pred_edges, thermal_edges)


class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1, lambda_structural=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        self.structural = StructuralConsistencyLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_structural = lambda_structural

    def forward(self, pred_rgb, target_rgb, thermal):
        l1_loss = self.l1(pred_rgb, target_rgb)
        perc_loss = self.perceptual(pred_rgb, target_rgb)
        struct_loss = self.structural(pred_rgb, thermal)

        total = (
            self.lambda_l1 * l1_loss
            + self.lambda_perceptual * perc_loss
            + self.lambda_structural * struct_loss
        )
        return total, {
            "l1": l1_loss.item(),
            "perceptual": perc_loss.item(),
            "structural": struct_loss.item(),
            "total": total.item(),
        }
