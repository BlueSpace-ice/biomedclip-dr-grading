# -*- coding: utf-8 -*-
"""
Misclassified case analysis & figure generation for Scientific Reports rebuttal.

Steps:
  1. Run inference on internal + external test sets via best.pt (image-only)
  2. Export all misclassified samples to misclassified_all.csv
  3. Select representative cases per reviewer-relevant error pattern (A–E)
  4. Generate a publication-quality figure (figure_misclassified.png / .pdf)
  5. Export selected_cases.csv for the authors' figure caption
"""

import os, sys, json, csv, warnings
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config as C
from model import load_biomedclip, DRGrader

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
H5_DIR    = C.DATA_DIR
CKPT_PATH = os.path.join(C.CKPT_DIR, "best.pt")
OUT_DIR   = C.RESULTS_DIR
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = C.DEVICE if torch.cuda.is_available() else "cpu"
SEED   = C.SEED

# error-type labels used in the figure & CSV
ERROR_TYPE_NAMES = {
    "A": "MildNPDR → SevNPDR (early-stage over-estimation)",
    "B": "PDR → MildNPDR / SevNPDR (high-risk missed PDR)",
    "C": "ModNPDR → Normal (dangerous missed pathology)",
    "D": "SevNPDR ↔ ModNPDR (adjacent-grade confusion)",
    "E": "Correct but ambiguous (boundary findings–image mismatch)",
}

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print("Loading BiomedCLIP backbone + best.pt …")
clip_model, preprocess_val, _, image_mean, image_std = load_biomedclip(C.MODEL_DIR, DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
print(f"  epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}, embed_dim={ckpt['embed_dim']}")

model = DRGrader(clip_model, ckpt["embed_dim"], C.NUM_CLASSES, freeze_text=True).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print("  model ready.\n")

# ---------------------------------------------------------------------------
# 2. Run inference & collect per-sample results
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_split(split_name, h5_path):
    """Return list of dicts, one per sample, including misclassified flag."""
    import h5py
    results = []
    f = h5py.File(h5_path, "r")
    keys = sorted(f["images"].keys(), key=lambda k: int(k))

    def _decode(x):
        """Decode bytes/object to clean string."""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        return str(x)

    for k in tqdm(keys, desc=f"infer {split_name}", unit="img"):
        arr  = f["images"][k][()]                     # (224,224,3) uint8
        y    = int(f["label"][k][()])
        patient   = _decode(f["patient"][k][()])
        fname     = _decode(f["filename"][k][()])
        findings  = _decode(f["findings_en"][k][()])
        label_name = _decode(f["label_name"][k][()])

        # preprocess image (same as BiomedCLIP eval transform)
        img = preprocess_val(Image.fromarray(arr)).unsqueeze(0).to(DEVICE)
        logits, _ = model(img)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]   # (5,)
        pred  = int(probs.argmax())
        conf  = float(probs.max())

        results.append({
            "split":       split_name,
            "key":         k,
            "patient":     patient,
            "filename":    fname,
            "y_true":      y,
            "y_pred":      pred,
            "y_true_name": C.DR_GRADES[y],
            "y_pred_name": C.DR_GRADES[pred],
            "confidence":  conf,
            "probs":       probs.tolist(),
            "findings_en": findings,
            "label_name":  label_name,
            "is_correct":  y == pred,
            "error_pair":  f"{C.DR_GRADES[y]}→{C.DR_GRADES[pred]}" if y != pred else "",
        })

    f.close()
    return results


for split, h5_key in [("internal_test", "internal_test"),
                       ("external_test", "external_test")]:
    cache_path = os.path.join(OUT_DIR, f"predictions_{split}.json")
    if os.path.exists(cache_path):
        print(f"Loading cached predictions for {split} …")
        with open(cache_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        # ensure float conversion back
        for r in existing:
            r["probs"] = [float(p) for p in r["probs"]]
            r["confidence"] = float(r["confidence"])
        if split == "internal_test":
            internal_all = existing
        else:
            external_all = existing
    else:
        results = infer_split(split, C.H5[h5_key])
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False)
        if split == "internal_test":
            internal_all = results
        else:
            external_all = results

# ---------------------------------------------------------------------------
# 3. Export ALL misclassified samples
# ---------------------------------------------------------------------------
print("\n--- Misclassification summary ---")
for label, all_results in [("internal", internal_all), ("external", external_all)]:
    n_total = len(all_results)
    n_correct = sum(1 for r in all_results if r["is_correct"])
    mis = [r for r in all_results if not r["is_correct"]]
    print(f"{label}: {n_correct}/{n_total} correct, {len(mis)} misclassified")

    # confusion-pair frequency
    pair_counts = defaultdict(int)
    for r in mis:
        pair_counts[r["error_pair"]] += 1
    print(f"  Top error pairs:")
    for pair, cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"    {pair}: {cnt}")

# write all misclassified to CSV
all_mis = [r for r in internal_all + external_all if not r["is_correct"]]
all_mis.sort(key=lambda r: -r["confidence"])
with open(os.path.join(OUT_DIR, "misclassified_all.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "split","key","patient","y_true_name","y_pred_name","confidence",
        "probs","error_pair","findings_en"])
    writer.writeheader()
    for r in all_mis:
        writer.writerow({k: r[k] for k in writer.fieldnames})
print(f"\nAll {len(all_mis)} misclassified → {OUT_DIR}/misclassified_all.csv")

# ---------------------------------------------------------------------------
# 4. Select representative cases (A–E), high-confidence first
# ---------------------------------------------------------------------------
mis_internal = [r for r in internal_all if not r["is_correct"]]
mis_external = [r for r in external_all if not r["is_correct"]]

# Error-type definitions: (true_grade_name, pred_grade_name)
# D only covers SevNPDR→ModNPDR (the classic adjacent downgrade)
ERROR_DEFS = {
    "A": [("MildNPDR", "SevNPDR")],                   # Mild→Severe: early-stage overestimation
    "B": [("PDR", "MildNPDR"), ("PDR", "SevNPDR")],   # PDR missed / under-estimated
    "C": [("ModNPDR", "Norm")],                        # Mod→Normal: dangerous missed pathology
    "D": [("SevNPDR", "ModNPDR")],                     # Severe→Moderate: adjacent-grade confusion
    # E handled separately: correct but low-confidence boundary case
}

# ---- Coordinated selection across both splits ----
# Strategy: each error type (A–D) contributes 1–2 cases, with at least one
# from each split. Internal gets 4–6, external 2–3. Total target ≈ 8 + 1 (E).
print("\n--- Coordinated case selection ---")
used_patients = set()
final_cases = []

def find_candidates(mis_pool, tname, pname, split_label=None):
    """Return misclassified cases matching (true, pred), optionally filtered by split."""
    out = [r for r in mis_pool
           if r["y_true_name"] == tname and r["y_pred_name"] == pname]
    if split_label:
        out = [r for r in out if r["split"] == split_label]
    out.sort(key=lambda r: -r["confidence"])
    return out

all_mis_pool = mis_internal + mis_external

for etype, pairs in ERROR_DEFS.items():
    # Collect from both splits
    intra_cands = []
    extra_cands = []
    for (tname, pname) in pairs:
        intra_cands.extend(find_candidates(mis_internal, tname, pname))
        extra_cands.extend(find_candidates(mis_external, tname, pname))

    intra_cands.sort(key=lambda r: -r["confidence"])
    extra_cands.sort(key=lambda r: -r["confidence"])

    # Pick 1 from internal and 1 from external (if available)
    picked_internal = None
    for c in intra_cands:
        if c["patient"] not in used_patients:
            picked_internal = c
            used_patients.add(c["patient"])
            break

    picked_external = None
    for c in extra_cands:
        if c["patient"] not in used_patients:
            picked_external = c
            used_patients.add(c["patient"])
            break

    reason = f"Internal: "
    if picked_internal:
        final_cases.append((etype, picked_internal))
        reason += f"conf={picked_internal['confidence']:.3f}"
    else:
        reason += f"none"
    reason += f" | External: "
    if picked_external:
        final_cases.append((etype, picked_external))
        reason += f"conf={picked_external['confidence']:.3f}"
    else:
        reason += f"none"
    print(f"  [{etype}] {reason}")

# ---- Pick E: one correct-but-ambiguous boundary case ----
correct_all = [r for r in internal_all + external_all if r["is_correct"]]
correct_all.sort(key=lambda r: r["confidence"])  # lowest confidence first
picked_e = None
for c in correct_all:
    if c["patient"] not in used_patients:
        picked_e = c
        used_patients.add(c["patient"])
        final_cases.append(("E", picked_e))
        print(f"  [E] picked: {c['split']} True={c['y_true_name']} conf={c['confidence']:.3f}")
        break

if not picked_e:
    print("  [E] No unused boundary case found; skipping")

print(f"\nTotal selected for figure: {len(final_cases)} cases")
for et, c in final_cases:
    print(f"  [{et}] {c['split']} | True={c['y_true_name']} Pred={c['y_pred_name']} "
          f"conf={c['confidence']:.3f} | patient hidden")

# ---------------------------------------------------------------------------
# 5. Export selected_cases.csv
# ---------------------------------------------------------------------------
csv_fields = ["error_type", "error_type_desc", "split", "key", "patient",
              "y_true", "y_true_name", "y_pred", "y_pred_name", "confidence",
              "probs", "top2_probs", "findings_en"]
with open(os.path.join(OUT_DIR, "selected_cases.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
    writer.writeheader()
    for et, c in final_cases:
        probs = c["probs"]
        # get top-2 indices
        sorted_idx = np.argsort(probs)[::-1]
        top2_str = (
            f"{C.DR_GRADES[sorted_idx[0]]} {probs[sorted_idx[0]]:.3f} / "
            f"{C.DR_GRADES[sorted_idx[1]]} {probs[sorted_idx[1]]:.3f}"
        )
        row = {
            **c,
            "error_type": et,
            "error_type_desc": ERROR_TYPE_NAMES.get(et, ""),
            "top2_probs": top2_str,
        }
        writer.writerow({k: row.get(k, "") for k in csv_fields})
print(f"selected_cases.csv → {OUT_DIR}/selected_cases.csv")

# ---------------------------------------------------------------------------
# 6. Generate publication-quality figure
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── style ──────────────────────────────────────────────────────────────────
TITLE_COLOR_ERR = "#C62828"
TITLE_COLOR_OK  = "#2E7D32"
TOP2_COLOR      = "#333333"
FINDINGS_COLOR  = "#777777"
BADGE_BG        = "#FFFFFF"

n_cases = len(final_cases)
N_COLS = 3
N_ROWS = (n_cases + N_COLS - 1) // N_COLS

# ── load H5 handles ────────────────────────────────────────────────────────
import h5py
h5_handles = {}
for et, c in final_cases:
    split = c["split"]
    if split not in h5_handles:
        h5_key = {"internal_test": "internal_test", "external_test": "external_test"}[split]
        h5_handles[split] = h5py.File(C.H5[h5_key], "r")

# ── create figure ──────────────────────────────────────────────────────────
# Each row: image (≈3.0 in) + title (≈0.4 in) + top2 (≈0.2 in) + findings (≈0.25 in)
ROW_H = 3.4
fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(13.5, ROW_H * N_ROWS + 0.5),
                         facecolor="white")
axes = axes.flatten() if n_cases > 1 else [axes]

# Manual spacing: tight hspace to reduce vertical gaps
plt.subplots_adjust(
    left=0.04, right=0.97,
    top=0.95, bottom=0.01,
    hspace=0.38, wspace=0.12,
)

# ── populate subplots ──────────────────────────────────────────────────────
for idx, (et, case) in enumerate(final_cases):
    ax = axes[idx]
    split = case["split"]
    key   = case["key"]

    # ── image ──────────────────────────────────────────────────────────
    arr = h5_handles[split]["images"][key][()]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        img_display = arr[:, :, 1]   # FFA green channel → grayscale
    else:
        img_display = arr

    ax.imshow(img_display, cmap="gray", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # thin grey frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#BBBBBB")
        spine.set_linewidth(0.6)

    # ── title: True | Pred (conf) ──────────────────────────────────────
    is_err = not case["is_correct"]
    color  = TITLE_COLOR_ERR if is_err else TITLE_COLOR_OK
    ax.set_title(
        f"True: {case['y_true_name']}  |  Pred: {case['y_pred_name']}"
        f"  (conf={case['confidence']:.2f})",
        color=color, fontsize=11, fontweight="bold", pad=6,
    )

    # ── Top-2: inside axes at the very bottom ──────────────────────────
    probs = case["probs"]
    order = np.argsort(probs)[::-1]
    top2_str = (
        f"{C.DR_GRADES[order[0]]} {probs[order[0]]:.3f}   /   "
        f"{C.DR_GRADES[order[1]]} {probs[order[1]]:.3f}"
    )
    ax.text(0.50, 0.010, top2_str, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=TOP2_COLOR,
            ha="center", va="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.85))

    # ── findings (xlabel slot, no overlap possible) ────────────────────
    findings = case["findings_en"]
    max_chars = 82
    if len(findings) > max_chars:
        findings = findings[:max_chars].rsplit(" ", 1)[0] + " …"
    ax.set_xlabel(findings, fontsize=7, color=FINDINGS_COLOR, labelpad=5)

    # ── error-type badge ───────────────────────────────────────────────
    ax.text(0.015, 0.985, f" [{et}] ", transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", color=color,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BADGE_BG,
                      edgecolor="#DDDDDD", alpha=0.92))

# ── hide unused ────────────────────────────────────────────────────────────
for idx in range(n_cases, len(axes)):
    axes[idx].set_visible(False)

# ── overall suptitle ───────────────────────────────────────────────────────
fig.suptitle("Representative Misclassified Cases — BiomedCLIP FFA DR Grading",
             fontsize=14, fontweight="bold", y=0.985)

# ── save ───────────────────────────────────────────────────────────────────
png_path = os.path.join(OUT_DIR, "figure_misclassified.png")
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nFigure saved → {png_path}")

try:
    pdf_path = os.path.join(OUT_DIR, "figure_misclassified.pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"PDF saved → {pdf_path}")
except Exception as e:
    print(f"PDF failed (non-critical): {e}")

# ── cleanup ────────────────────────────────────────────────────────────────
for fh in h5_handles.values():
    try:
        fh.close()
    except Exception:
        pass
plt.close("all")
print("\nDone.")
