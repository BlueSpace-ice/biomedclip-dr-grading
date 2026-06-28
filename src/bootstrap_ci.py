# -*- coding: utf-8 -*-
"""
Patient-level cluster bootstrap 95% confidence intervals for 5-class DR grading,
plus paired significance testing (multimodal vs image-only).

Key design (addressing reviewer comment "no CIs, no statistical tests"):

1. Inference runs ONCE per model per split → saves (y, pred, prob, patient).
2. Bootstrap resamples PATIENTS (not individual images), because multiple images
   from the same patient (adjacent key-frames, both eyes) are NOT independent.
   Resampling images would underestimate variance → CIs too narrow.
3. B=1000 iterations. Each iteration draws K patients with replacement (K = number
   of unique patients in the split), collects all their images, computes all metrics.
4. For each metric: point estimate = full-sample value; 95% CI = [2.5%, 97.5%]
   percentile of the bootstrap distribution. np.nanpercentile handles degenerate
   resamples where a class is missing → AUROC can't be computed.
5. Paired test: both models share the SAME resampled patient indices each iteration.
   Δ = multimodal_metric − imageonly_metric. Report mean Δ, 95% CI of Δ, and
   approximate two-sided bootstrap p-value: p ≈ 2 × min(frac(Δ>0), frac(Δ<0)).

Output (saved to F:\FFA_h5_dataset\results\):
  - metrics_{split}_ci.json   — point estimate + CI for every metric
  - metrics_{split}_ci.csv    — same data in tabular form, formatted "0.571 (0.540–0.602)"
  - paired_test_{split}.json  — paired comparison results
  - paired_test_{split}.csv   — same in tabular form

Run:  python bootstrap_ci.py
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import torch
import h5py
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, accuracy_score)

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config as C
from model import load_biomedclip, DRGrader
from dataset import make_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# 0.  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

B = 1000          # bootstrap replicates
CI_ALPHA = 0.05   # 95% CI → [2.5%, 97.5%]
CI_LO, CI_HI = 100 * CI_ALPHA / 2, 100 * (1 - CI_ALPHA / 2)  # (2.5, 97.5)

CLASS_NAMES = C.DR_GRADES           # ["Norm","MildNPDR","ModNPDR","SevNPDR","PDR"]
N_CLASSES = C.NUM_CLASSES

# metrics for paired significance testing
PAIRED_METRICS = ["overall_accuracy", "weighted_f1"] \
               + [f"sensitivity_{c}" for c in CLASS_NAMES]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, device):
    """Load BiomedCLIP + DRGrader from a checkpoint. Returns (model, embed_dim)."""
    clip_model, _, _, mean, std = load_biomedclip(C.MODEL_DIR, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    embed_dim = ckpt["embed_dim"]
    model = DRGrader(clip_model, embed_dim, N_CLASSES, C.FREEZE_TEXT).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, mean, std, ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan"))


@torch.no_grad()
def infer(model, loader, device):
    """Run image-only inference. Returns (y, pred, prob)."""
    model.eval()
    ys, ps, probs = [], [], []
    for images, labels in tqdm(loader, desc="  infer", unit="batch", leave=False):
        images = images.to(device)
        logits, _ = model(images)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        ps.append(p.argmax(1))
        ys.append(labels.numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(probs))


def load_patient_ids(h5_path):
    """Load patient IDs in the SAME order as FFADataset (sorted integer keys)."""
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["images"].keys(), key=lambda k: int(k))
        pids = []
        for k in keys:
            raw = f["patient"][k][()]
            if isinstance(raw, bytes):
                pids.append(raw.decode())
            else:
                pids.append(str(raw))
    return np.array(pids)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Metric computation (matches evaluate.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def per_class_specificity(cm):
    """TN / (TN + FP) per class. Matches evaluate.py:per_class_specificity."""
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
    """One-vs-rest accuracy per class. Matches evaluate.py:per_class_accuracy."""
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
    """Compute the full metrics dict for one bootstrap replicate.

    Returns dict with keys like overall_accuracy, weighted_f1, and per-class
    acc/precision/sensitivity/specificity/auroc. Returns None for degenerate
    cases (empty y).
    """
    if len(y) == 0:
        return None

    cm = confusion_matrix(y, pred, labels=list(range(N_CLASSES)))

    spec = per_class_specificity(cm)
    pcacc = per_class_accuracy_one_vs_rest(cm)
    prec = precision_score(y, pred, average=None, labels=range(N_CLASSES), zero_division=0)
    sens = recall_score(y, pred, average=None, labels=range(N_CLASSES), zero_division=0)

    # one-vs-rest AUROC per class — may fail if only one class present
    auroc = []
    for i in range(N_CLASSES):
        yi = (y == i).astype(int)
        # need at least two classes present for AUROC
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
        result[f"accuracy_{name}"] = float(pcacc[i])
        result[f"precision_{name}"] = float(prec[i])
        result[f"sensitivity_{name}"] = float(sens[i])
        result[f"specificity_{name}"] = float(spec[i])
        result[f"auroc_{name}"] = float(auroc[i])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Patient-level cluster bootstrap core
# ═══════════════════════════════════════════════════════════════════════════════

def build_patient_blocks(patient_ids):
    """Group sample indices by patient.

    Returns
    -------
    blocks : list of np.ndarray
        blocks[p] = array of sample indices belonging to patient p.
    unique_patients : np.ndarray
        Unique patient IDs (strings), length = len(blocks).
    """
    unique_pids, inverse = np.unique(patient_ids, return_inverse=True)
    blocks = [np.where(inverse == p)[0] for p in range(len(unique_pids))]
    return blocks, unique_pids


def bootstrap_replicate(blocks, n_patients, y, pred, prob):
    """One bootstrap replicate: resample patients with replacement.

    Parameters
    ----------
    blocks : list of np.ndarray
        Each entry is the sample indices for one patient.
    n_patients : int
        Number of unique patients → draw this many with replacement.
    y, pred, prob : np.ndarray
        Full-sample inference results.

    Returns
    -------
    metrics dict or None if degenerate.
    """
    # draw patient indices with replacement
    idx = np.random.randint(0, len(blocks), size=n_patients)
    # collect all samples from selected patients
    sample_indices = np.concatenate([blocks[i] for i in idx])
    return compute_all_metrics(y[sample_indices], pred[sample_indices],
                               prob[sample_indices])


def bootstrap_ci_single(y, pred, prob, patient_ids, label=""):
    """Patient-level cluster bootstrap for ONE model.

    Returns
    -------
    point : dict
        Point estimates (full sample, no resampling).
    ci : dict
        {metric_name: (lo, hi)} 95% CI.
    all_reps : list of dict
        All B bootstrap metric dicts (for paired comparison).
    """
    blocks, unique_pids = build_patient_blocks(patient_ids)
    n_patients = len(blocks)

    print(f"  {label}: {len(y)} images, {n_patients} patients "
          f"(avg {len(y)/n_patients:.1f} img/pat)")

    # point estimate on full sample
    point = compute_all_metrics(y, pred, prob)
    metric_names = list(point.keys())

    # bootstrap
    reps = []
    for _ in tqdm(range(B), desc=f"  bootstrap {label}", unit="rep", leave=False):
        rep = bootstrap_replicate(blocks, n_patients, y, pred, prob)
        reps.append(rep)

    # aggregate
    ci = {}
    for m in metric_names:
        vals = np.array([r[m] if (r is not None and m in r) else np.nan
                         for r in reps], dtype=np.float64)
        ci[m] = (float(np.nanpercentile(vals, CI_LO)),
                 float(np.nanpercentile(vals, CI_HI)))

    return point, ci, reps


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Paired significance test
# ═══════════════════════════════════════════════════════════════════════════════

def paired_test(reps_a, reps_b, metric_names):
    """Paired bootstrap test: both models share the same resampled patient draws.

    Parameters
    ----------
    reps_a, reps_b : list of dict
        Bootstrap replicates from model A (multimodal) and B (image-only).
        Must come from the SAME set of patient draws (same random state).
    metric_names : list of str
        Metrics to test.

    Returns
    -------
    dict: {metric: {mean_delta, ci_lo, ci_hi, frac_positive, p_approx}}
    """
    results = {}
    for m in metric_names:
        deltas = []
        for ra, rb in zip(reps_a, reps_b):
            if ra is None or rb is None or m not in ra or m not in rb:
                deltas.append(np.nan)
                continue
            va = ra[m]
            vb = rb[m]
            if np.isnan(va) or np.isnan(vb):
                deltas.append(np.nan)
            else:
                deltas.append(va - vb)
        deltas = np.array(deltas, dtype=np.float64)
        valid = deltas[~np.isnan(deltas)]
        if len(valid) == 0:
            results[m] = {"mean_delta": float("nan"), "ci_lo": float("nan"),
                          "ci_hi": float("nan"), "frac_>0": float("nan"),
                          "p_approx": float("nan")}
            continue

        frac_pos = np.mean(valid > 0)
        frac_neg = np.mean(valid < 0)
        # two-sided bootstrap p ≈ 2 × min(P(Δ>0), P(Δ<0))
        # (conservative: use max of the two tails → min fraction × 2)
        p_approx = 2.0 * min(frac_pos, frac_neg)
        results[m] = {
            "mean_delta": float(np.mean(valid)),
            "ci_lo": float(np.nanpercentile(deltas, CI_LO)),
            "ci_hi": float(np.nanpercentile(deltas, CI_HI)),
            "frac_>0": float(frac_pos),
            "p_approx": float(p_approx),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Main evaluation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_with_bootstrap(split, model_mm, model_io, mean, std, device):
    """Run both models on one split, then bootstrap + paired test."""
    h5_path = C.H5[split]
    patient_ids = load_patient_ids(h5_path)
    loader = make_loader(split, mean, std, shuffle=False)

    # ── Inference (once per model) ──────────────────────────────────────────
    print(f"\n{'='*70}\n  Split: {split}\n{'='*70}")

    print("  [1/4] Inference: multimodal model (best.pt) ...")
    t0 = time.time()
    y_mm, pred_mm, prob_mm = infer(model_mm, loader, device)
    print(f"        done in {time.time()-t0:.1f}s")

    # Re-create loader (it was exhausted)
    loader2 = make_loader(split, mean, std, shuffle=False)
    print("  [2/4] Inference: image-only model (best_imageonly.pt) ...")
    t0 = time.time()
    y_io, pred_io, prob_io = infer(model_io, loader2, device)
    print(f"        done in {time.time()-t0:.1f}s")

    # ── Bootstrap (same seed for both → same patient draws) ─────────────────
    print(f"\n  [3/4] Patient-level cluster bootstrap (B={B}) ...")
    # Reset seed so the two bootstrap runs use identical random draws
    np.random.seed(SEED)
    point_mm, ci_mm, reps_mm = bootstrap_ci_single(
        y_mm, pred_mm, prob_mm, patient_ids, label="multimodal")

    np.random.seed(SEED)   # reset → paired comparison is valid
    point_io, ci_io, reps_io = bootstrap_ci_single(
        y_io, pred_io, prob_io, patient_ids, label="imageonly")

    # ── Paired test ─────────────────────────────────────────────────────────
    print(f"\n  [4/4] Paired significance test ...")
    paired = paired_test(reps_mm, reps_io, PAIRED_METRICS)

    return {
        "split": split,
        "n_images": int(len(y_mm)),
        "n_patients": int(len(np.unique(patient_ids))),
        "multimodal": {"point": point_mm, "ci": ci_mm},
        "imageonly":   {"point": point_io, "ci": ci_io},
        "paired_test": paired,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Output formatters
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_ci(point, ci_lo, ci_hi, decimals=3):
    """Format as '0.571 (0.540–0.602)'."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(point)} ({fmt.format(ci_lo)}-{fmt.format(ci_hi)})"


def build_flat_table(point, ci):
    """Build a flat dict {metric_name: formatted_string} for CSV output."""
    rows = []
    # overall
    for m in ["overall_accuracy", "weighted_f1"]:
        p = point[m]
        lo, hi = ci[m]
        rows.append({"metric": m, "value": fmt_ci(p, lo, hi),
                     "point": p, "ci_lo": lo, "ci_hi": hi})
    # per-class
    for c in CLASS_NAMES:
        for m in ["accuracy", "precision", "sensitivity", "specificity", "auroc"]:
            key = f"{m}_{c}"
            p = point[key]
            lo, hi = ci[key]
            rows.append({"metric": key, "value": fmt_ci(p, lo, hi),
                         "point": p, "ci_lo": lo, "ci_hi": hi})
    return rows


def save_results(result, out_dir):
    """Save JSON + CSV for metrics and paired test."""
    split = result["split"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Multimodal metrics ──────────────────────────────────────────────────
    mm_rows = build_flat_table(result["multimodal"]["point"],
                               result["multimodal"]["ci"])
    with open(os.path.join(out_dir, f"metrics_{split}_ci.json"), "w",
              encoding="utf-8") as f:
        json.dump(mm_rows, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, f"metrics_{split}_ci.csv"), "w",
              encoding="utf-8") as f:
        f.write("metric,value\n")
        for r in mm_rows:
            f.write(f"{r['metric']},{r['value']}\n")

    # ── Image-only metrics ──────────────────────────────────────────────────
    io_rows = build_flat_table(result["imageonly"]["point"],
                               result["imageonly"]["ci"])
    with open(os.path.join(out_dir, f"metrics_{split}_imageonly_ci.json"), "w",
              encoding="utf-8") as f:
        json.dump(io_rows, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, f"metrics_{split}_imageonly_ci.csv"), "w",
              encoding="utf-8") as f:
        f.write("metric,value\n")
        for r in io_rows:
            f.write(f"{r['metric']},{r['value']}\n")

    # ── Paired test ─────────────────────────────────────────────────────────
    paired = result["paired_test"]
    with open(os.path.join(out_dir, f"paired_test_{split}.json"), "w",
              encoding="utf-8") as f:
        json.dump(paired, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, f"paired_test_{split}.csv"), "w",
              encoding="utf-8") as f:
        f.write("metric,mean_delta,ci_lo,ci_hi,p_approx,significant(p<0.05)\n")
        for m, d in paired.items():
            sig = "Yes" if d["p_approx"] < 0.05 else "No"
            f.write(f"{m},{d['mean_delta']:.4f},{d['ci_lo']:.4f},"
                    f"{d['ci_hi']:.4f},{d['p_approx']:.4f},{sig}\n")

    return mm_rows, paired


def print_summary(result, mm_rows, paired):
    """Print a readable summary table to the terminal."""
    split = result["split"]
    print(f"\n{'='*70}")
    print(f"  RESULTS: {split}")
    print(f"  n={result['n_images']} images, {result['n_patients']} patients")
    print(f"{'='*70}")

    print(f"\n  --- Multimodal (best.pt) point estimates + 95% CI ---")
    print(f"  {'metric':<28}{'value':>22}")
    print(f"  {'-'*50}")
    for r in mm_rows:
        print(f"  {r['metric']:<28}{r['value']:>22}")

    print(f"\n  --- Paired test: Multimodal - ImageOnly ---")
    print(f"  {'metric':<28}{'d_mean':>8}{'95% CI':>20}{'p~':>8}{'sig':>6}")
    print(f"  {'-'*70}")
    for m, d in paired.items():
        ci_str = f"({d['ci_lo']:.4f}-{d['ci_hi']:.4f})"
        sig = " *" if d["p_approx"] < 0.05 else ""
        print(f"  {m:<28}{d['mean_delta']:>8.4f}{ci_str:>20}"
              f"{d['p_approx']:>8.4f}{sig:>6}")
    print(f"\n  * p < 0.05 (approximate two-sided bootstrap p-value)")

    # Count significant
    n_sig = sum(1 for d in paired.values() if d["p_approx"] < 0.05)
    print(f"  {n_sig}/{len(paired)} metrics significant at α=0.05")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Bootstrap replicates: B = {B}")
    print(f"Random seed: {SEED}")

    # ── Load models ─────────────────────────────────────────────────────────
    print("\n── Loading models ──")
    ckpt_mm = os.path.join(C.CKPT_DIR, "best.pt")
    ckpt_io = os.path.join(C.CKPT_DIR, "best_imageonly.pt")

    model_mm, mean, std, ep_mm, vl_mm = load_model(ckpt_mm, device)
    print(f"  best.pt           epoch={ep_mm}, val_loss={vl_mm:.4f}")

    model_io, _, _, ep_io, vl_io = load_model(ckpt_io, device)
    print(f"  best_imageonly.pt epoch={ep_io}, val_loss={vl_io:.4f}")

    # ── Evaluate both splits ────────────────────────────────────────────────
    all_results = {}
    for split in ["internal_test", "external_test"]:
        result = evaluate_with_bootstrap(split, model_mm, model_io, mean, std, device)
        mm_rows, paired = save_results(result, C.RESULTS_DIR)
        print_summary(result, mm_rows, paired)
        all_results[split] = result

    # ── Final summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  All results saved to: {C.RESULTS_DIR}")
    print(f"    metrics_*_ci.json / .csv       — point estimates + 95% CI")
    print(f"    paired_test_*.json / .csv       — paired significance tests")
    print(f"{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
