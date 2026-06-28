# -*- coding: utf-8 -*-
"""
Train an ImageNet-pretrained baseline (ResNet-50 or ViT-B/16) for DR grading.

Identical experimental conditions to the main BiomedCLIP model:
  - Same train/val H5 splits (patient-level, not re-divided)
  - Same BATCH_SIZE, EPOCHS, LR, GRAD_CLIP_NORM, SEED
  - Same class weights, early-stopping patience, LR scheduler
  - Same optimizer (AdamW) and ReduceLROnPlateau
  - Same validation metric for checkpoint selection (lowest val_loss)
  - Only difference: backbone source (ImageNet vs BiomedCLIP) + no contrastive loss

Usage:
  python train_baseline.py --backbone resnet50             --tag resnet50
  python train_baseline.py --backbone vit_base_patch16_224 --tag imagenet_vit
"""
import os, sys, random, argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from model_baseline import create_baseline, IMAGENET_MEAN, IMAGENET_STD
from dataset import make_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):  return x


# ── reproducibility ──────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ── class weights (same as train.py) ────────────────────────────────────────
def compute_class_weights(h5_path, num_classes):
    import h5py
    with h5py.File(h5_path, "r") as f:
        counts = Counter()
        for k in f["label"]:
            counts[int(f["label"][k][()])] += 1
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        weights[c] = total / (num_classes * max(cnt, 1))
    return weights


# ── validation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, ce):
    model.eval()
    tot, n, correct = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        tot += ce(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return tot / n, correct / n


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True,
                        choices=["resnet50", "vit_base_patch16_224"])
    parser.add_argument("--tag", required=True,
                        help="Short name for checkpoint & results, e.g. resnet50")
    args = parser.parse_args()

    set_seed(C.SEED)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Backbone: {args.backbone}  |  tag: {args.tag}  |  device: {device}")

    # ── verify identical config ───────────────────────────────────────────────
    print(f"Config check: BATCH={C.BATCH_SIZE} EPOCHS={C.EPOCHS} LR={C.LR} "
          f"GRAD_CLIP={C.GRAD_CLIP_NORM} CLASS_W={C.USE_CLASS_WEIGHTS} SEED={C.SEED} "
          f"EARLY_STOP={C.EARLY_STOP_PATIENCE} LR_PATIENCE={C.LR_SCHEDULER_PATIENCE} "
          f"LR_FACTOR={C.LR_SCHEDULER_FACTOR}")
    print(f"Normalization: mean={IMAGENET_MEAN} std={IMAGENET_STD}")
    print(f"Loss: CrossEntropy only (no contrastive)")

    # ── model ─────────────────────────────────────────────────────────────────
    model, feat_dim = create_baseline(args.backbone, C.NUM_CLASSES, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.1f}M  |  feature dim: {feat_dim}")

    # ── data (ImageNet normalization) ─────────────────────────────────────────
    train_loader = make_loader("train", IMAGENET_MEAN, IMAGENET_STD, shuffle=True)
    val_loader   = make_loader("val",   IMAGENET_MEAN, IMAGENET_STD, shuffle=False)
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── class weights ─────────────────────────────────────────────────────────
    if C.USE_CLASS_WEIGHTS:
        class_w = compute_class_weights(C.H5["train"], C.NUM_CLASSES).to(device)
        print(f"Class weights: {class_w.tolist()}")
    else:
        class_w = None
    ce = nn.CrossEntropyLoss(weight=class_w)

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=C.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=C.LR_SCHEDULER_FACTOR,
        patience=C.LR_SCHEDULER_PATIENCE, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(C.USE_AMP and device == "cuda"))

    # ── training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")
    best_epoch = 1
    ckpt_path = os.path.join(C.CKPT_DIR, f"best_{args.tag}.pt")
    early_stop_counter = 0

    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{C.EPOCHS}", unit="batch")
        for batch in pbar:
            images, labels = batch   # baseline: no tokens
            images, labels = images.to(device), labels.to(device)

            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=(C.USE_AMP and device == "cuda")):
                logits = model(images)
                loss = ce(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(params, C.GRAD_CLIP_NORM)
            scaler.step(optim)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(ce=f"{loss.item():.3f}",
                             lr=f"{optim.param_groups[0]['lr']:.2e}")

        val_loss, val_acc = validate(model, val_loader, device, ce)
        cur_lr = optim.param_groups[0]["lr"]
        print(f"  epoch {epoch}: train_loss={running/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={cur_lr:.2e}")

        # checkpoint + early stop
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({
                "model": model.state_dict(),
                "backbone": args.backbone,
                "feat_dim": feat_dim,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  saved best → {ckpt_path} (val_loss={val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= C.EARLY_STOP_PATIENCE:
                print(f"  early stop: no improvement for {C.EARLY_STOP_PATIENCE} epochs")
                break

        scheduler.step(val_loss)

    print(f"\nDone. Best val_loss={best_val:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Next: python evaluate_baseline.py --tag {args.tag}")


if __name__ == "__main__":
    main()
