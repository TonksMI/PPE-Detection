# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Goal

Three-assignment computer vision pipeline for industrial PPE compliance:

1. **Assignment 1** — Crop-level PPE classification (5 classes, MinhNKB schema)
2. **Assignment 2** — End-to-end object detection on full scenes (YOLOv8n)
3. **Assignment 3** — Pixel-level segmentation (keremberke 10-class schema)

## Two Schemas — Do Not Mix

### MinhNKB 5-class (Assignments 1 & 2)
Whole-person labels: `helmet`, `safety_vest`, `full_ppe`, `partial_ppe`, `no_ppe`
- Source: `D:\Claude\datasets\helmet-safety-vest-detection-master\`
- No safety glasses, gloves, or footwear in this schema

### Keremberke 10-class (Assignment 3 only)
Individual item labels: `helmet`, `no_helmet`, `glove`, `no_glove`, `goggles`, `no_goggles`, `mask`, `no_mask`, `shoes`, `no_shoes`
- Source: `train.zip`, `test.zip`, `valid.zip` in project root (keremberke HF dataset)
- No safety vests, full_ppe, or partial_ppe in this schema

## Commands

```bash
# Train core CNN + ML ensemble (Assignment 1)
python src/ppe_production_train.py --epochs 100 --batch-size 256 --max-per-class 600

# Fine-tune ViT-B/16 (Assignment 1 best)
python src/ppe_vit_train.py

# Two-stage inference: YOLO person detect → CNN classify
python src/ppe_combined_pipeline.py

# CCTV sliding-window validation
python src/ppe_cctv_validation.py

# Build keremberke 10-class segmentation dataset (SAM2, ~100 min for 4000 images)
python src/ppe_seg_keremberke_rebuild.py

# Train DeepLabV3+ semantic segmentation (Assignment 3)
python src/ppe_deeplab_train.py

# Train YOLOv8n-seg instance segmentation (Assignment 3)
python src/ppe_yolo_seg_train.py

# Multi-model comparison chart
python src/ppe_experiment_comparison.py

# Generate full evaluation report (ROC, confusion matrices, PR curves)
python src/ppe_full_evaluation_report.py

# Fine-tune YOLOv8n person detector
yolo train data=person_detection/person_detect.yaml model=yolov8n.pt epochs=10 imgsz=416 freeze=9

# Generate Word document reports
python reports/generate_assignment_report.py        # Final project report
cd reports && node gen_assignment3.js               # Assignment 3 report
```

## Architecture

### Assignment 1 Data Flow

```
D:\datasets\jomarkow\                  (YOLO format, head annotations)
D:\datasets\helmet-safety-vest-detection-master\  (Pascal VOC XML)
D:\datasets\inria_person\              (pedestrian crops)
        ↓
setup_datasets.py                      downloads and organises
        ↓
src/ppe_production_train.py            XML → person crops → cache
        ↓
PPENetFast (CNN)  → results/models/prod_cnn_model.pth
SVM/RF/ET/GBM    → results/models/prod_*.pkl  (gitignored, large)
        ↓
src/ppe_combined_pipeline.py           YOLOv8n → crops → PPENetFast
```

### Assignment 3 Data Flow

```
train.zip / test.zip / valid.zip       (keremberke COCO format, project root)
        ↓
src/ppe_seg_keremberke_rebuild.py      SAM2 box prompts → pixel masks
        ↓
D:\Claude\datasets\ppe_seg_ke\         semantic + instance dataset
        ↓
src/ppe_deeplab_train.py               DeepLabV3+ ResNet50 (11 classes)
src/ppe_yolo_seg_train.py              YOLOv8n-seg (10 classes)
        ↓
results/models/deeplab_model.pth       (gitignored — >100MB)
results/models/yolo_seg_best.pt
```

## Key Results

| Model | Schema | Metric | Value |
|---|---|---|---|
| ViT-B/16 (fine-tuned) | MinhNKB 5-class | Val Accuracy | **93.90%** |
| PPENetFast CNN | MinhNKB 5-class | Val Accuracy | 87.33% |
| YOLOv8n end-to-end | MinhNKB 5-class | mAP50 | 86.3% |
| YOLOv8n-seg | keremberke 10-class | Box/Mask mAP50 | **90.0% / 87.1%** |
| DeepLabV3+ ResNet50 | keremberke 10-class | mIoU | pending |
| SVM (RBF, PCA-220) | MinhNKB 5-class | Val Accuracy | 76.31% |

## PPENetFast Architecture (`src/ppe_cnn_fast.py`)

3 conv blocks (32→64→128 filters), AdaptiveAvgPool(2,2), FC 512→256→5, 226K params.
Trained with OneCycleLR max_lr=1e-3, label_smoothing=0.05, batch=256.

## Path Resolution

All scripts auto-detect Windows vs Linux:
```python
for _cand in ["D:/datasets", "D:/Claude/datasets",
              str(Path(__file__).resolve().parents[2] / "datasets")]:
    if os.path.exists(os.path.join(_cand, "jomarkow")):
        DATASETS = _cand
        break
```

Segmentation dataset uses `ppe_seg_ke/` (keremberke 10-class) not `ppe_seg/` (legacy MinhNKB).
Cache lives at `D:\Claude\cache\` (`.npy` files are gitignored).

## Environment

- Python 3.10.6, PyTorch 2.12.0.dev+cu128, CUDA 12.8
- GPU: RTX 5070 (12GB) — batch=256 safe for CNN; batch=8 safe for DeepLabV3+
- `C:\Program Files\PyManager\python3.exe` — use this executable

## Git / Storage Notes

- `.pkl` files globally gitignored (ML model files too large)
- `.pth` / `.pt` neural network weights tracked except those >100MB
- `results/models/prod_vit_model.pth` and `results/models/deeplab_model.pth` gitignored
- Large files stripped from history with `git filter-repo` when needed
- Commit style: `feat:`, `results:`, `fix:`, `docs:`
- New scripts → `src/`, plots → `results/models/`, weights → `results/models/`
- Keremberke zips (`*.zip` in root) are gitignored — too large for GitHub
