# -*- coding: utf-8 -*-
"""
Evaluate the fine-tuned model on the internal and external test sets.
Inference is IMAGE-ONLY (text discarded), matching the paper.

Metrics per class + overall: accuracy, precision, sensitivity (recall),
specificity, weighted F1, one-vs-rest AUROC. Saves a JSON + confusion matrix
CSV per split into results/.

Run:  python evaluate.py
"""
import os
import sys
import json

import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, accuracy_score)

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from model import load_biomedclip, DRGrader
from dataset import make_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for images, labels in tqdm(loader, desc="infer", unit="batch"):
        images = images.to(device)
        logits, _ = model(images)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        ps.append(p.argmax(1))
        ys.append(labels.numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(probs))


def per_class_specificity(cm):
    spec = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return spec


def per_class_accuracy(cm):
    """one-vs-rest accuracy per class (matches the paper's per-grade 'accuracy')."""
    accs = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        accs.append((tp + tn) / total)
    return accs


def evaluate_split(model, split, mean, std, device):
    loader = make_loader(split, mean, std, shuffle=False)
    y, pred, prob = infer(model, loader, device)
    cm = confusion_matrix(y, pred, labels=list(range(C.NUM_CLASSES)))

    spec = per_class_specificity(cm)
    pcacc = per_class_accuracy(cm)
    prec = precision_score(y, pred, average=None, labels=range(C.NUM_CLASSES), zero_division=0)
    sens = recall_score(y, pred, average=None, labels=range(C.NUM_CLASSES), zero_division=0)
    # one-vs-rest AUROC per class
    auroc = []
    for i in range(C.NUM_CLASSES):
        yi = (y == i).astype(int)
        try:
            auroc.append(roc_auc_score(yi, prob[:, i]))
        except ValueError:
            auroc.append(float("nan"))

    result = {
        "split": split,
        "n": int(len(y)),
        "overall_accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "per_class": {
            C.DR_GRADES[i]: {
                "accuracy": float(pcacc[i]),
                "precision": float(prec[i]),
                "sensitivity": float(sens[i]),
                "specificity": float(spec[i]),
                "auroc": float(auroc[i]),
            } for i in range(C.NUM_CLASSES)
        },
    }
    # save
    with open(os.path.join(C.RESULTS_DIR, f"metrics_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savetxt(os.path.join(C.RESULTS_DIR, f"confusion_{split}.csv"), cm,
               fmt="%d", delimiter=",",
               header=",".join(C.DR_GRADES), comments="")
    return result, cm


def print_result(res, cm):
    print(f"\n=== {res['split']}  (n={res['n']}) ===")
    print(f"overall_accuracy={res['overall_accuracy']:.4f}  "
          f"weighted_F1={res['weighted_f1']:.4f}")
    print(f"{'grade':<10}{'acc':>8}{'prec':>8}{'sens':>8}{'spec':>8}{'auroc':>8}")
    for g in C.DR_GRADES:
        d = res["per_class"][g]
        print(f"{g:<10}{d['accuracy']:>8.3f}{d['precision']:>8.3f}"
              f"{d['sensitivity']:>8.3f}{d['specificity']:>8.3f}{d['auroc']:>8.3f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("   " + " ".join(f"{g[:5]:>6}" for g in C.DR_GRADES))
    for i, g in enumerate(C.DR_GRADES):
        print(f"{g[:5]:>5} " + " ".join(f"{cm[i, j]:>6d}" for j in range(C.NUM_CLASSES)))


def main():
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    clip_model, _, _, mean, std = load_biomedclip(C.MODEL_DIR, device)
    ckpt = torch.load(os.path.join(C.CKPT_DIR, "best_imageonly.pt"), map_location=device)
    model = DRGrader(clip_model, ckpt["embed_dim"], C.NUM_CLASSES, C.FREEZE_TEXT).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded best_imageonly.pt (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f})")

    for split in ("internal_test", "external_test"):
        res, cm = evaluate_split(model, split, mean, std, device)
        print_result(res, cm)
    print(f"\nMetrics + confusion matrices saved to {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
