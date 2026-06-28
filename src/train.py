# -*- coding: utf-8 -*-
"""
Fine-tune BiomedCLIP for DR grading.

Training loss = class-weighted CE(classification) + LAMBDA * cross-modal contrastive.
  * contrastive loss masks same-text pairs (templated reports → no false negatives).
  * early-stopping + ReduceLROnPlateau so we don't waste compute after the first epoch.
Text encoder frozen; ViT image encoder + linear head trained end-to-end.
Best checkpoint = lowest validation loss (val is image-only -> CE only).

Run:  python train.py
"""
import os
import sys
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from model import load_biomedclip, DRGrader, contrastive_loss
from dataset import make_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def compute_class_weights(h5_path, num_classes):
    """Return 1D tensor of class weights (balanced: 1 / frequency)."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        labels = f["label"]
        counts = Counter()
        for k in labels:
            counts[int(labels[k][()])] += 1
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        weights[c] = total / (num_classes * max(cnt, 1))
    return weights


def build_same_text_mask(tokens):
    """tokens: (B, L) LongTensor.  Returns (B, B) BoolTensor: True iff text_i == text_j."""
    # (1, B, L) == (B, 1, L) → (B, B, L) → all along last dim → (B, B)
    return (tokens.unsqueeze(0) == tokens.unsqueeze(1)).all(dim=-1)


@torch.no_grad()
def validate(model, loader, device, ce):
    model.eval()
    tot, n, correct = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        tot += ce(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return tot / n, correct / n


def main():
    set_seed(C.SEED)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    clip_model, _, tokenizer, mean, std = load_biomedclip(C.MODEL_DIR, device)
    # infer embedding dim from the image projection
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        embed_dim = clip_model.encode_image(dummy).shape[-1]
    model = DRGrader(clip_model, embed_dim, C.NUM_CLASSES, C.FREEZE_TEXT).to(device)

    train_loader = make_loader("train", mean, std, tokenizer=tokenizer)
    val_loader = make_loader("val", mean, std, shuffle=False)

    # ---- class weights --------------------------------------------------
    if C.USE_CLASS_WEIGHTS:
        class_w = compute_class_weights(C.H5["train"], C.NUM_CLASSES).to(device)
        print(f"class_weights={class_w.tolist()}")
    else:
        class_w = None
    ce = nn.CrossEntropyLoss(weight=class_w)

    # ---- optimiser & scheduler ------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.1f}M")
    optim = torch.optim.AdamW(params, lr=C.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=C.LR_SCHEDULER_FACTOR,
        patience=C.LR_SCHEDULER_PATIENCE, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(C.USE_AMP and device == "cuda"))

    best_val = float("inf")
    best_epoch = 1
    best_path = os.path.join(C.CKPT_DIR, "best_imageonly.pt")
    early_stop_counter = 0

    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{C.EPOCHS}", unit="batch")
        for batch in pbar:
            images, labels, tokens = batch
            images, labels = images.to(device), labels.to(device)
            tokens = tokens.to(device)

            # ---- same-text mask for contrastive loss -----------------------
            same_text_mask = build_same_text_mask(tokens) if C.LAMBDA_CONTRASTIVE > 0 else None

            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=(C.USE_AMP and device == "cuda")):
                logits, img_feats = model(images)
                loss_ce = ce(logits, labels)
                with torch.no_grad() if C.FREEZE_TEXT else torch.enable_grad():
                    txt_feats = model.encode_text(tokens)
                loss_con = contrastive_loss(img_feats, txt_feats, model.logit_scale,
                                            text_mask=same_text_mask)
                loss = loss_ce + C.LAMBDA_CONTRASTIVE * loss_con
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(params, C.GRAD_CLIP_NORM)
            scaler.step(optim)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(ce=f"{loss_ce.item():.3f}", con=f"{loss_con.item():.3f}",
                             lr=f"{optim.param_groups[0]['lr']:.2e}")

        val_loss, val_acc = validate(model, val_loader, device, ce)
        cur_lr = optim.param_groups[0]["lr"]
        print(f"  epoch {epoch}: train_loss={running/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={cur_lr:.2e}")

        # ---- early-stopping & checkpoint logic ----------------------------
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({"model": model.state_dict(), "embed_dim": embed_dim,
                        "epoch": epoch, "val_loss": val_loss}, best_path)
            print(f"  saved best -> {best_path} (val_loss={val_loss:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= C.EARLY_STOP_PATIENCE:
                print(f"  early stop: no improvement for {C.EARLY_STOP_PATIENCE} epochs")
                break

        scheduler.step(val_loss)

    print(f"\nDone. Best val_loss={best_val:.4f} at epoch {best_epoch}. Checkpoint: {best_path}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
