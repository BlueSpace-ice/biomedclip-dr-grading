# FFA DR Grading — BiomedCLIP fine-tuning

Reproduces the manuscript pipeline: fine-tune BiomedCLIP on paired FFA
image / English clinical-findings text, using text as a cross-modal
regularizer during training and discarding it at inference (image-only).

## Layout
```
F:\FFA_h5_dataset\
├─ dr_train.h5 / dr_val.h5 / dr_internal_test.h5 / dr_external_test.h5
├─ models\biomedclip\        <- put the HuggingFace download here
├─ src\  config.py dataset.py model.py train.py evaluate.py
├─ checkpoints\  results\    <- auto-created
└─ requirements.txt
```

## 1. Environment
```bash
# install torch matching your CUDA first (example: cu118)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 2. Download BiomedCLIP weights (once, on a machine with internet)
```bash
huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
  --local-dir F:/FFA_h5_dataset/models/biomedclip
```
The folder must end up containing `open_clip_config.json`,
`open_clip_pytorch_model.bin`, and the tokenizer files. `model.py` loads
everything from this local folder (no internet needed at train time).

## 3. Train
```bash
cd F:\FFA_h5_dataset\src
python train.py
```
- Loss = cross-entropy + cross-modal contrastive (text encoder frozen).
- Best checkpoint (lowest val loss) -> `checkpoints/best.pt`.
- Hyper-params (batch 16, 15 epochs, lr 1e-5, grad-clip 1.0) are in `config.py`.

## 4. Evaluate
```bash
python evaluate.py
```
Image-only inference on internal + external test. Writes per-split metrics
(`results/metrics_*.json`) and confusion matrices (`results/confusion_*.csv`):
per-class accuracy, precision, sensitivity, specificity, weighted-F1, AUROC.

## Notes
- **Text field**: training uses `findings_en` (clinical findings) as the
  semantic anchor. `diagnosis_en` is intentionally NOT used — it paraphrases
  the label and would weaken the contrastive signal.
- **Inference is unimodal**: val/test load images only; no text is read.
- **Templated reports**: many `findings_en` strings are identical across
  samples, so in-batch contrastive has some false negatives. To reduce this,
  you can mask same-text pairs in `contrastive_loss` (treat identical-text
  pairs as positives) — left out by default to match the paper's plain setup.
- **Windows + num_workers**: if you hit a multiprocessing error, set
  `NUM_WORKERS = 0` in `config.py`.
- **GPU**: training the ViT-B/16 on 5,000 images for 15 epochs is light; any
  modern GPU (or your H200 node) is fine. Set `DEVICE="cpu"` to debug.
```
