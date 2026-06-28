# -*- coding: utf-8 -*-
"""
Generate baseline comparison table (CSV + Markdown) for the rebuttal.

Reads existing CI JSONs for all four models, extracts overall_accuracy and
weighted_f1 (with 95% CI), and writes a side-by-side comparison table.

Models:
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. ResNet-50            ImageNet pretrain,  image-only         │
  │ 2. ViT-B/16 (ImageNet)  ImageNet pretrain,  image-only         │
  │ 3. ViT-B/16 (image-only)BiomedCLIP pretrain, image-only         │
  │ 4. ViT-B/16 (proposed)  BiomedCLIP pretrain, multimodal (CE+CL) │
  └─────────────────────────────────────────────────────────────────┘

Usage:
  python compare_baselines.py
    -- assumes all CI JSONs already exist in results/
"""
import os, sys, json

sys.path.insert(0, os.path.dirname(__file__))
import config as C


# ── model definitions ────────────────────────────────────────────────────────
MODELS = [
    {
        "name": "ResNet-50",
        "pretrain": "ImageNet",
        "text_in_training": False,
        "ci_key_internal": "metrics_resnet50_internal_test_ci.json",
        "ci_key_external": "metrics_resnet50_external_test_ci.json",
    },
    {
        "name": "ViT-B/16 (ImageNet)",
        "pretrain": "ImageNet",
        "text_in_training": False,
        "ci_key_internal": "metrics_imagenet_vit_internal_test_ci.json",
        "ci_key_external": "metrics_imagenet_vit_external_test_ci.json",
    },
    {
        "name": "ViT-B/16 (image-only)",
        "pretrain": "BiomedCLIP",
        "text_in_training": False,
        "ci_key_internal": "metrics_internal_test_imageonly_ci.json",
        "ci_key_external": "metrics_external_test_imageonly_ci.json",
    },
    {
        "name": "ViT-B/16 (proposed)",
        "pretrain": "BiomedCLIP",
        "text_in_training": True,
        "ci_key_internal": "metrics_internal_test_ci.json",
        "ci_key_external": "metrics_external_test_ci.json",
    },
]


def load_ci(filepath):
    """Load a {metric_name → {point, ci_lo, ci_hi}} dict from a CI JSON file."""
    if not os.path.exists(filepath):
        print(f"  ⚠ missing: {filepath}")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        out[r["metric"]] = {
            "point": r["point"],
            "ci_lo": r["ci_lo"],
            "ci_hi": r["ci_hi"],
            "formatted": r.get("value", ""),
        }
    return out


def fmt(val, ci_lo, ci_hi, decimals=3):
    """Format as '0.571 (0.504–0.636)'."""
    fmt_str = f"{{:.{decimals}f}}"
    return f"{fmt_str.format(val)} ({fmt_str.format(ci_lo)}–{fmt_str.format(ci_hi)})"


def main():
    print("Building baseline comparison table …\n")

    # ── collect data ──────────────────────────────────────────────────────────
    rows = []
    for m in MODELS:
        ci_int = load_ci(os.path.join(C.RESULTS_DIR, m["ci_key_internal"]))
        ci_ext = load_ci(os.path.join(C.RESULTS_DIR, m["ci_key_external"]))

        int_acc = ci_int.get("overall_accuracy", {})
        int_wf1 = ci_int.get("weighted_f1", {})
        ext_acc = ci_ext.get("overall_accuracy", {})
        ext_wf1 = ci_ext.get("weighted_f1", {})

        if not int_acc:
            int_acc_str, int_wf1_str = "—", "—"
        else:
            int_acc_str = fmt(int_acc["point"], int_acc["ci_lo"], int_acc["ci_hi"])
            int_wf1_str = fmt(int_wf1["point"], int_wf1["ci_lo"], int_wf1["ci_hi"])

        if not ext_acc:
            ext_acc_str, ext_wf1_str = "—", "—"
        else:
            ext_acc_str = fmt(ext_acc["point"], ext_acc["ci_lo"], ext_acc["ci_hi"])
            ext_wf1_str = fmt(ext_wf1["point"], ext_wf1["ci_lo"], ext_wf1["ci_hi"])

        rows.append({
            "Model": m["name"],
            "Pretrain": m["pretrain"],
            "Text": "Yes" if m["text_in_training"] else "No",
            "Internal Acc (95% CI)": int_acc_str,
            "Internal wF1 (95% CI)": int_wf1_str,
            "External Acc (95% CI)": ext_acc_str,
            "External wF1 (95% CI)": ext_wf1_str,
        })

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(C.RESULTS_DIR, "baseline_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Model,Pretrain,Text-in-Training,"
                "Internal Acc (95% CI),Internal wF1 (95% CI),"
                "External Acc (95% CI),External wF1 (95% CI)\n")
        for r in rows:
            f.write(f"{r['Model']},{r['Pretrain']},{r['Text']},"
                    f"\"{r['Internal Acc (95% CI)']}\",\"{r['Internal wF1 (95% CI)']}\","
                    f"\"{r['External Acc (95% CI)']}\",\"{r['External wF1 (95% CI)']}\"\n")
    print(f"CSV → {csv_path}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    md_path = os.path.join(C.RESULTS_DIR, "baseline_comparison.md")
    lines = []
    lines.append("# DR Grading — Model Comparison (95% CI)")
    lines.append("")
    lines.append("| Model | Pretrain | Text | Internal Acc (95% CI) | Internal wF1 (95% CI) | External Acc (95% CI) | External wF1 (95% CI) |")
    lines.append("|-------|----------|------|----------------------|----------------------|----------------------|----------------------|")
    for r in rows:
        lines.append(f"| {r['Model']} | {r['Pretrain']} | {r['Text']} | "
                     f"{r['Internal Acc (95% CI)']} | {r['Internal wF1 (95% CI)']} | "
                     f"{r['External Acc (95% CI)']} | {r['External wF1 (95% CI)']} |")
    lines.append("")
    lines.append(f"*All metrics computed identically (patient-level bootstrap, B=1000, seed=42).*")
    lines.append(f"*Training conditions: BATCH=16, EPOCHS=15, LR=1e-5, AdamW, class-balanced CE, early-stop patience=5.*")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"MD  → {md_path}")

    # ── print terminal ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    for r in rows:
        print(f"  {r['Model']:<28} {r['Pretrain']:<12} Text={r['Text']:<3}"
              f"  Int Acc={r['Internal Acc (95% CI)']:<28}"
              f"  Ext Acc={r['External Acc (95% CI)']}")
    print("=" * 90)
    print("Done.")


if __name__ == "__main__":
    main()
