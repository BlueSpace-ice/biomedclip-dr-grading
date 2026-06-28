# -*- coding: utf-8 -*-
"""
Baseline models for DR grading — ImageNet-pretrained ResNet-50 & ViT-B/16.

Each baseline = timm backbone (pretrained, num_classes=0 → pooled features)
+ nn.Linear head.  No text, no contrastive loss — pure image classifiers.

Usage (via train_baseline.py):
    python train_baseline.py --backbone resnet50             --tag resnet50
    python train_baseline.py --backbone vit_base_patch16_224 --tag imagenet_vit
"""
import torch
import torch.nn as nn
import timm

# Standard ImageNet normalization (torchvision / timm default)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class BaselineGrader(nn.Module):
    """timm backbone → pooled features → linear classifier."""

    def __init__(self, backbone_name, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0)
        # Infer feature dimension
        self.feat_dim = self._infer_feat_dim()
        self.head = nn.Linear(self.feat_dim, num_classes)

    def _infer_feat_dim(self):
        """Run a dummy forward to get feature dimension."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            return feats.shape[-1]

    def forward(self, images):
        feats = self.backbone(images)
        logits = self.head(feats)
        return logits


def create_baseline(backbone_name, num_classes=5, device="cpu"):
    """Factory: returns (BaselineGrader, feat_dim) on the requested device."""
    model = BaselineGrader(backbone_name, num_classes=num_classes)
    model = model.to(device)
    return model, model.feat_dim
