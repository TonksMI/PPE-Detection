# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Two-stage computer vision pipeline for industrial PPE compliance:
1. **Person detection** — YOLOv8n fine-tuned on hard-hat worker images
2. **PPE classification** — CNN + ML ensemble on person crops

5 classes: `helmet`, `safety_vest`, `full_ppe`, `partial_ppe`, `no_ppe`

## Commands

```bash
# Train all production models (CNN + SVM/RF/ET/GBM ensemble)
python src/ppe_production_train.py --epochs 100 --batch-size 256 --max-per-class 600

# Run two-stage inference pipeline (YOLO person detect → CNN classify)
python src/ppe_combined_pipeline.py

# CCTV sliding-window validation
python src/ppe_cctv_validation.py

# Generate full evaluation report (ROC curves, confusion matrices, PR curves)
python src/ppe_full_evaluation_report.py

# Fine-tune YOLOv8n person detector
yolo train data=person_detection/person_detect.yaml model=yolov8n.pt epochs=10 imgsz=416 freeze=9

# Generate Word document report
cd reports && node create_prod_report.js

# Setup/download datasets
python setup_datasets.py
```

## Architecture

### Data Flow

```
D:\datasets\jomarkow\          (YOLO format, head annotations)
D:\datasets\helmet-safety-vest-detection-master\  (Pascal VOC XML)
D:\datasets\inria_person\      (pedestrian crops for YOLO training)
        ↓
setup_datasets.py              downloads and organises the above
        ↓
src/ppe_production_train.py    parses XML/YOLO annotations → crops persons
                               → D:\Claude\cache\crops_X_600.npy (cache, git-ignored)
        ↓
PPENetFast (32×32 CNN)    → results/models/prod_cnn_model.pth
SVM / RF / ET / GBM       → results/models/prod_*.pkl (git-ignored, too large)
        ↓
src/ppe_combined_pipeline.py   YOLOv8n → person crops → PPENetFast classify
```

### Key Models & Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| CNN PPENet (100 epochs) | **87.33%** | `prod_cnn_model.pth` — tracked in git |
| SVM (PCA 220 + RBF, balanced) | 76.31% | `prod_svm_multi.pkl` — git-ignored |
| YOLOv8n fine-tuned | mAP50=0.679 | `person_detector_yolov8n.pt` — tracked |

Weakest classes: `partial_ppe` F1=0.76, `full_ppe` F1=0.77.

### PPENetFast Architecture (`src/ppe_cnn_fast.py`)

3 conv blocks (32→64→128 filters), AdaptiveAvgPool(2,2), FC 512→256→5, 226K params.
Trained with OneCycleLR max_lr=1e-3, label_smoothing=0.05, batch=256.

### Path Resolution (all scripts)

Scripts auto-detect Windows vs Linux:
```python
if os.path.exists("D:/datasets/jomarkow"):
    DATASETS = "D:/datasets"
else:
    DATASETS = os.path.join(BASE, "datasets")  # BASE = parent of PPE-Detection/
```
Cache lives at `D:\Claude\cache\` (`.npy` files are git-ignored).

## Environment

- Python 3.10.6, PyTorch 2.12.0.dev+cu128, CUDA 12.8
- GPU: RTX 5070 (12GB) — batch=256 is safe; use `device=0` for YOLO
- Datasets: `D:\datasets\` — not in repo
- Cache: `D:\Claude\cache\` — not in repo

## Git / Storage Notes

- `.pkl` files are globally git-ignored (ML models are too large). Only `.pth` and `.pt` neural network weights are tracked.
- `*.npy` dataset cache files are git-ignored.
- Commit style: `feat:`, `results:`, `fix:`, `docs:`
- New scripts → `src/`, plots → `results/plots/`, models → `results/models/`

## Priorities for Improvement

1. More YOLOv8n fine-tuning epochs (4 done, mAP50=0.679; target 0.85+)
2. Transfer learning for PPE classifier (MobileNetV2 or EfficientNet-B0)
3. Data augmentation (currently minimal)
4. Class weighting for `partial_ppe` / `full_ppe` (underrepresented)
5. End-to-end YOLO for PPE detection (single-stage)
