# Fine-tuned BiomedCLIP for DR Grading

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

BiomedCLIP fine-tuned on fluorescein angiography (FFA) images with English clinical-findings text as a cross-modal regularizer.
5-class diabetic retinopathy (DR) grading: Norm, MildNPDR, ModNPDR, SevNPDR, PDR.

## Key Features

- **Multimodal training** — cross-entropy + cross-modal contrastive loss; text encoder frozen, ViT image encoder fine-tuned
- **Image-only inference** — text discarded at test time, matching clinical deployment
- **Two baselines** — ResNet-50 & ViT-B/16 (ImageNet pretrained) under identical experimental conditions
- **Patient-level cluster bootstrap** — 95% CIs that respect within-patient correlation (B=1000)
- **Misclassification figure generator** — publication-quality figure for rebuttals (Scientific Reports)

## Project Structure

```
├── src/
│   ├── config.py                     # All paths & hyperparameters
│   ├── model.py                      # BiomedCLIP loader + DRGrader
│   ├── model_baseline.py             # BaselineGrader (timm backbones)
│   ├── dataset.py                    # H5 dataloader
│   ├── train.py                      # Train BiomedCLIP (CE + contrastive)
│   ├── train_baseline.py             # Train baseline (CE only)
│   ├── evaluate.py                   # Metrics: acc/precision/sensitivity/specificity/F1/AUROC
│   ├── evaluate_baseline.py          # Evaluate baselines + bootstrap CI
│   ├── bootstrap_ci.py               # Patient-level cluster bootstrap + paired tests
│   ├── misclassification_analysis.py # Error-case selection + rebuttal figure
│   └── compare_baselines.py          # Four-model comparison table (CSV + MD)
├── requirements.txt
└── .gitignore
```

Data and weights are NOT included — download them separately:
- **FFA H5 files**: `dr_train.h5`, `dr_val.h5`, `dr_internal_test.h5`, `dr_external_test.h5`
- **BiomedCLIP weights**: `models/biomedclip/` (from HuggingFace)

## Quick Start

### 1. Environment
```bash
conda create -n ffa python=3.9 -y && conda activate ffa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Download BiomedCLIP
```bash
huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --local-dir models/biomedclip
```

### 3. Train
```bash
cd src

# Main model (BiomedCLIP + contrastive)
python train.py

# Baselines (identical config, CE only)
python train_baseline.py --backbone resnet50             --tag resnet50
python train_baseline.py --backbone vit_base_patch16_224 --tag imagenet_vit
```

### 4. Evaluate
```bash
# Main model
python evaluate.py
python bootstrap_ci.py

# Baselines (with bootstrap CI)
python evaluate_baseline.py --tag resnet50
python evaluate_baseline.py --tag imagenet_vit

# Comparison table
python compare_baselines.py
```

### 5. Misclassification Analysis (Rebuttal Figure)
```bash
python misclassification_analysis.py
# → results/figure_misclassified.png (.pdf)
# → results/selected_cases.csv
```

## Notes

- **Text field**: training uses `findings_en` (clinical findings), not `diagnosis_en` (would leak labels)
- **Inference is unimodal**: text only used during training as a regularizer
- **Config**: edit `src/config.py` to change batch size, LR, epochs, etc.
- **Windows**: set `NUM_WORKERS = 0` in `config.py` if multiprocessing fails

## Citation

```bibtex
@misc{biomedclip-dr-grading,
  title     = {Fine-tuned BiomedCLIP for DR Grading},
  author    = {},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/BlueSpace-ice/biomedclip-dr-grading}
}
```
