# -*- coding: utf-8 -*-
"""
Evaluate a trained baseline model (ResNet-50 / ViT-B/16) on internal + external test sets.

Metrics computed IDENTICALLY to evaluate.py:
  - per-class: one-vs-rest accuracy, precision, sensitivity, specificity, AUROC
  - overall:   accuracy, weighted F1
  - confusion matrix

Also runs patient-level cluster bootstrap (B=1000, seed=42) for 95% CIs,
using the same algorithm as bootstrap_ci.py.

Usage:
  python evaluate_baseline.py --tag resnet50
  python evaluate_baseline.py --tag imagenet_vit --backbone vit_base_patch16_224
"""
import os, sys, json, time, argparse, warnings
import numpy as np
import torch
import h5py
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, accuracy_score,
)

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from model_baseline import BaselineGrader, IMAGENET_MEAN, IMAGENET_STD
from dataset import make_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):  return x

warnings.filterwarnings("ignore")

# ── constants ────────────────────────────────────────────────────────────────
SEED = 42
B = 1000
CI_ALPHA = 0.05
CI_LO, CI_HI = 100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)
CLASS_NAMES = C.DR_GRADES
N_CLASSES = C.NUM_CLASSES


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Metric functions (exact copies from evaluate.py / bootstrap_ci.py)
# ═══════════════════════════════════════════════════════════════════════════════

def per_class_specificity(cm):
    spec = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.array(spec)


def per_class_accuracy_one_vs_rest(cm):
    accs = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        accs.append((tp + tn) / total)
    return np.array(accs)


def compute_all_metrics(y, pred, prob):
    if len(y) == 0:
        return None
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    spec = per_class_specificity(cm)
    pcacc = per_class_accuracy_one_vs_rest(cm)
    prec = precision_score(y, pred, average=None, labels=range(N_CLASSES), zero_division=0)
    sens = recall_score(y, pred, average=None, labels=range(N_CLASSES), zero_division=0)
    auroc = []
    for i in range(N_CLASSES):
        yi = (y == i).astype(int)
        if len(np.unique(yi)) < 2:
            auroc.append(float("nan"))
            continue
        try:
            auroc.append(roc_auc_score(yi, prob[:, i]))
        except ValueError:
            auroc.append(float("nan"))
    result = {
        "overall_accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    for i, name in enumerate(CLASS_NAMES):
        result[f"accuracy_{name}"]   = float(pcacc[i])
        result[f"precision_{name}"]  = float(prec[i])
        result[f"sensitivity_{name}"] = float(sens[i])
        result[f"specificity_{name}"] = float(spec[i])
        result[f"auroc_{name}"]       = float(auroc[i])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Inference
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for images, labels in tqdm(loader, desc="  infer", unit="batch", leave=False):
        images = images.to(device)
        logits = model(images)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        ps.append(p.argmax(1))
        ys.append(labels.numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(probs))


def load_patient_ids(h5_path):
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["images"].keys(), key=lambda k: int(k))
        pids = []
        for k in keys:
            raw = f["patient"][k][()]
            pids.append(raw.decode() if isinstance(raw, bytes) else str(raw))
    return np.array(pids)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Patient-level cluster bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

def build_patient_blocks(patient_ids):
    unique_pids, inverse = np.unique(patient_ids, return_inverse=True)
    blocks = [np.where(inverse == p)[0] for p in range(len(unique_pids))]
    return blocks, unique_pids


def bootstrap_replicate(blocks, n_patients, y, pred, prob):
    idx = np.random.randint(0, len(blocks), size=n_patients)
    sample_indices = np.concatenate([blocks[i] for i in idx])
    return compute_all_metrics(y[sample_indices], pred[sample_indices], prob[sample_indices])


def bootstrap_ci_single(y, pred, prob, patient_ids, label=""):
    blocks, unique_pids = build_patient_blocks(patient_ids)
    n_patients = len(blocks)
    print(f"  {label}: {len(y)} images, {n_patients} patients "
          f"(avg {len(y) / n_patients:.1f} img/pat)")

    point = compute_all_metrics(y, pred, prob)
    metric_names = list(point.keys())

    reps = []
    for _ in tqdm(range(B), desc=f"  bootstrap {label}", unit="rep", leave=False):
        rep = bootstrap_replicate(blocks, n_patients, y, pred, prob)
        reps.append(rep)

    ci = {}
    for m in metric_names:
        vals = np.array([r[m] if (r is not None and m in r) else np.nan
                         for r in reps], dtype=np.float64)
        ci[m] = (float(np.nanpercentile(vals, CI_LO)),
                 float(np.nanpercentile(vals, CI_HI)))
    return point, ci, reps


def fmt_ci(point, ci_lo, ci_hi, decimals=3):
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(point)} ({fmt.format(ci_lo)}-{fmt.format(ci_hi)})"


def build_flat_table(point, ci):
    rows = []
    for m in ["overall_accuracy", "weighted_f1"]:
        p = point[m]
        lo, hi = ci[m]
        rows.append({"metric": m, "value": fmt_ci(p, lo, hi),
                     "point": p, "ci_lo": lo, "ci_hi": hi})
    for c in CLASS_NAMES:
        for m in ["accuracy", "precision", "sensitivity", "specificity", "auroc"]:
            key = f"{m}_{c}"
            p = point[key]
            lo, hi = ci[key]
            rows.append({"metric": key, "value": fmt_ci(p, lo, hi),
                         "point": p, "ci_lo": lo, "ci_hi": hi})
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_split(model, split, tag, device):
    h5_path = C.H5[split]
    patient_ids = load_patient_ids(h5_path)
    loader = make_loader(split, IMAGENET_MEAN, IMAGENET_STD, shuffle=False)

    t0 = time.time()
    y, pred, prob = infer(model, loader, device)
    print(f"  inference done in {time.time() - t0:.1f}s")

    np.random.seed(SEED)
    point, ci, _ = bootstrap_ci_single(y, pred, prob, patient_ids, label=f"{tag}/{split}")

    # plain confusion matrix
    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))
    np.savetxt(os.path.join(C.RESULTS_DIR, f"confusion_{tag}_{split}.csv"), cm,
               fmt="%d", delimiter=",", header=",".join(CLASS_NAMES), comments="")

    # plain metrics (no CI)
    plain = {
        "split": split,
        "n": int(len(y)),
        "overall_accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "per_class": {
            CLASS_NAMES[i]: {
                "accuracy":    float(point[f"accuracy_{CLASS_NAMES[i]}"]),
                "precision":   float(point[f"precision_{CLASS_NAMES[i]}"]),
                "sensitivity": float(point[f"sensitivity_{CLASS_NAMES[i]}"]),
                "specificity": float(point[f"specificity_{CLASS_NAMES[i]}"]),
                "auroc":       float(point[f"auroc_{CLASS_NAMES[i]}"]),
            } for i in range(N_CLASSES)
        },
    }
    with open(os.path.join(C.RESULTS_DIR, f"metrics_{tag}_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(plain, f, ensure_ascii=False, indent=2)

    # CI metrics
    ci_rows = build_flat_table(point, ci)
    with open(os.path.join(C.RESULTS_DIR, f"metrics_{tag}_{split}_ci.json"), "w", encoding="utf-8") as f:
        json.dump(ci_rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(C.RESULTS_DIR, f"metrics_{tag}_{split}_ci.csv"), "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for r in ci_rows:
            f.write(f"{r['metric']},{r['value']}\n")

    return point, ci, y, pred, prob


def print_result(tag, split, point, ci):
    print(f"\n{'='*60}")
    print(f"  {tag}  |  {split}  (n_images=... see JSON)")
    acc = point["overall_accuracy"]
    wf1 = point["weighted_f1"]
    print(f"  overall_accuracy  = {fmt_ci(acc, *ci['overall_accuracy'])}")
    print(f"  weighted_f1       = {fmt_ci(wf1, *ci['weighted_f1'])}")
    print(f"  {'grade':<10}{'acc':>18}{'prec':>18}{'sens':>18}{'spec':>18}{'auroc':>18}")
    for c in CLASS_NAMES:
        print(f"  {c:<10}"
              f"{fmt_ci(point[f'accuracy_{c}'],   *ci[f'accuracy_{c}']):>18}"
              f"{fmt_ci(point[f'precision_{c}'],  *ci[f'precision_{c}']):>18}"
              f"{fmt_ci(point[f'sensitivity_{c}'], *ci[f'sensitivity_{c}']):>18}"
              f"{fmt_ci(point[f'specificity_{c}'], *ci[f'specificity_{c}']):>18}"
              f"{fmt_ci(point[f'auroc_{c}'],       *ci[f'auroc_{c}']):>18}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="e.g. resnet50 / imagenet_vit")
    parser.add_argument("--backbone", default=None,
                        help="default: inferred from tag (resnet50→resnet50, imagenet_vit→vit_base_patch16_224)")
    args = parser.parse_args()

    # infer backbone name
    if args.backbone:
        backbone = args.backbone
    elif "imagenet_vit" in args.tag or "vit" in args.tag:
        backbone = "vit_base_patch16_224"
    else:
        backbone = "resnet50"

    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(C.CKPT_DIR, f"best_{args.tag}.pt")
    print(f"Loading {args.tag}  |  backbone={backbone}  |  device={device}")
    print(f"Checkpoint: {ckpt_path}")

    # ── load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    feat_dim = ckpt.get("feat_dim")
    model = BaselineGrader(backbone, C.NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"  epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', '?'):.4f}, "
          f"feat_dim={feat_dim}")

    # ── evaluate ──────────────────────────────────────────────────────────────
    for split in ["internal_test", "external_test"]:
        point, ci, _, _, _ = evaluate_split(model, split, args.tag, device)
        print_result(args.tag, split, point, ci)

    print(f"\nAll results → {C.RESULTS_DIR}/metrics_{args.tag}_*.json  "
          f"+ confusion_{args.tag}_*.csv")


if __name__ == "__main__":
    main()
