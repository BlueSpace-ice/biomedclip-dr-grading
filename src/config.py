# -*- coding: utf-8 -*-
"""Central configuration. Edit paths/hyper-params here only."""
import os

# ---- paths (default: project lives in F:\FFA_h5_dataset) -------------------
PROJECT_DIR = os.environ.get("FFA_PROJECT_DIR", r"F:\FFA_h5_dataset")
DATA_DIR = PROJECT_DIR                                   # the dr_*.h5 files
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "biomedclip")   # HF download
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

H5 = {
    "train":         os.path.join(DATA_DIR, "dr_train.h5"),
    "val":           os.path.join(DATA_DIR, "dr_val.h5"),
    "internal_test": os.path.join(DATA_DIR, "dr_internal_test.h5"),
    "external_test": os.path.join(DATA_DIR, "dr_external_test.h5"),
}

DR_GRADES = ["Norm", "MildNPDR", "ModNPDR", "SevNPDR", "PDR"]
NUM_CLASSES = 5

# ---- training (from manuscript section 2.4) -------------------------------
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-5
GRAD_CLIP_NORM = 1.0
MAX_TEXT_LEN = 256          # PubMedBERT context length
LAMBDA_CONTRASTIVE = 0.0    # 消融: 纯图像 baseline（对比损失关闭）
TEXT_FIELD = "findings_en"  # semantic anchor = clinical findings (NOT diagnosis_en)
FREEZE_TEXT = True          # paper: text encoder frozen, ViT fine-tuned
USE_AMP = True              # mixed precision
USE_CLASS_WEIGHTS = True    # balanced cross-entropy to rescue minority classes (esp. Mild)

# ---- early stopping & LR schedule -------------------------------------------
EARLY_STOP_PATIENCE = 5      # stop if no val_loss improvement for N epochs
LR_SCHEDULER_PATIENCE = 3    # ReduceLROnPlateau: halve LR after N stagnant epochs
LR_SCHEDULER_FACTOR = 0.5

SEED = 42
NUM_WORKERS = 4             # set 0 on Windows if you hit multiprocessing issues
DEVICE = "cuda"            # "cuda" or "cpu"

for d in (CKPT_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
