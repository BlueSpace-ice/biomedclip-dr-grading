# -*- coding: utf-8 -*-
"""
Load BiomedCLIP (HuggingFace weights, open_clip architecture) from a LOCAL
directory, add a classification head, and freeze the text tower.

The HF repo `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` ships an
`open_clip_config.json` (model_cfg + preprocess_cfg) and `open_clip_pytorch_model.bin`.
We register that config and build the model fully offline.
"""
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import _MODEL_CONFIGS

_LOCAL_NAME = "biomedclip_ffa"


def load_biomedclip(model_dir, device="cpu"):
    """Returns (clip_model, preprocess_val, tokenizer, image_mean, image_std)."""
    cfg_path = os.path.join(model_dir, "open_clip_config.json")
    ckpt_path = os.path.join(model_dir, "open_clip_pytorch_model.bin")
    if not os.path.exists(cfg_path) or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Expected open_clip_config.json and open_clip_pytorch_model.bin in {model_dir}. "
            f"Download with:\n  huggingface-cli download "
            f"microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 --local-dir {model_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_cfg = cfg["model_cfg"]
    pp = cfg.get("preprocess_cfg", {})

    # Point the HF text tokenizer/config at the LOCAL dir so nothing is downloaded.
    # (tokenizer files ship inside the same repo folder.)
    model_cfg.setdefault("text_cfg", {})
    model_cfg["text_cfg"]["hf_tokenizer_name"] = model_dir

    # Register so open_clip knows this custom architecture, then load weights.
    _MODEL_CONFIGS[_LOCAL_NAME] = model_cfg
    image_mean = tuple(pp.get("mean", (0.48145466, 0.4578275, 0.40821073)))
    image_std = tuple(pp.get("std", (0.26862954, 0.26130258, 0.27577711)))

    model, _, preprocess_val = create_model_and_transforms(
        _LOCAL_NAME,
        pretrained=ckpt_path,
        image_mean=image_mean,
        image_std=image_std,
    )
    tokenizer = get_tokenizer(_LOCAL_NAME)
    model = model.to(device)
    return model, preprocess_val, tokenizer, image_mean, image_std


class DRGrader(nn.Module):
    """BiomedCLIP image encoder + linear head. Text tower used only in training
    (for the cross-modal contrastive loss) and frozen; discarded at inference."""

    def __init__(self, clip_model, embed_dim, num_classes, freeze_text=True):
        super().__init__()
        self.clip = clip_model
        self.head = nn.Linear(embed_dim, num_classes)
        if freeze_text:
            for name, p in self.clip.named_parameters():
                # freeze everything on the text side; keep visual + logit_scale trainable
                if name.startswith("text") or name.startswith("transformer") \
                        or "token_embedding" in name or "positional_embedding" in name \
                        or name.startswith("ln_final") or name == "text_projection":
                    p.requires_grad_(False)

    def encode_image(self, images):
        return self.clip.encode_image(images)          # (B, D) projected embedding

    def encode_text(self, tokens):
        return self.clip.encode_text(tokens)           # (B, D) projected embedding

    def forward(self, images):
        feats = self.encode_image(images)
        logits = self.head(feats)
        return logits, feats

    @property
    def logit_scale(self):
        return self.clip.logit_scale


def contrastive_loss(image_feats, text_feats, logit_scale, text_mask=None):
    """Standard symmetric CLIP InfoNCE within the batch.

    Parameters
    ----------
    text_mask : BoolTensor (B, B) or None
        mask[i][j] == True  →  text_i == text_j (same clinical-findings string).
        Non‑diagonal True entries are set to -inf so templated duplicates do NOT
        act as false negatives. This directly addresses the con-loss floor issue.
    """
    img = F.normalize(image_feats, dim=-1)
    txt = F.normalize(text_feats, dim=-1)
    scale = logit_scale.exp().clamp(max=100.0)
    logits_img = scale * img @ txt.t()
    logits_txt = logits_img.t()          # symmetric before masking

    if text_mask is not None:
        # everything that shares the same text but is NOT the diagonal
        neg_mask = text_mask & ~torch.eye(text_mask.size(0), dtype=torch.bool,
                                          device=text_mask.device)
        logits_img = logits_img.masked_fill(neg_mask, float("-inf"))
        logits_txt = logits_txt.masked_fill(neg_mask.t(), float("-inf"))

    labels = torch.arange(img.size(0), device=img.device)
    return 0.5 * (F.cross_entropy(logits_img, labels)
                  + F.cross_entropy(logits_txt, labels))
