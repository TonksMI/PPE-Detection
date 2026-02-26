# PPE Detection — Computer Vision Final Project

Automated Personal Protective Equipment (PPE) detection for industrial warehouse environments using a two-stage computer vision pipeline.

## Overview

| Stage | Method | Result |
|-------|--------|--------|
| Person Detection | YOLOv8n fine-tuned | mAP50 = 0.679 |
| PPE Classification | CNN PPENet — 100 epochs, RTX 5070 | **87.33% multi-class** |
| Binary (PPE / no PPE) | SVM (HOG + colour features) | **84.44%** |
| Best Multi-class AUC | CNN PPENet (one-vs-rest) | **0.978** |

### Pipeline

```
CCTV Frame → YOLOv8n Person Detector → Person Crops → CNN PPENet → PPE Classification
```

**Stage 1 — Person Detection:** YOLOv8n fine-tuned on 1,105 images (jomarkow hard-hat workers with head-to-body box expansion + 300 INRIA pedestrian crops). Pre-trained on COCO (118K images).

**Stage 2 — PPE Classification:** Custom lightweight CNN (207K parameters, 64×64 input) classifying 5 safety classes with 87.33% validation accuracy.

---

## Classes

| Class | Description |
|-------|-------------|
| `helmet` | Safety hard hat worn correctly |
| `safety_vest` | High-visibility reflective vest |
| `full_ppe` | Both helmet and vest present |
| `partial_ppe` | Some PPE present but incomplete |
| `no_ppe` | No safety equipment detected |

---

## Datasets

| Dataset | Images | Format | Use |
|---------|--------|--------|-----|
| [MinhNKB Helmet-Safety-Vest](https://github.com/MinhNKB/helmet-safety-vest-detection) | 1,613 | Pascal VOC XML | PPE classification training |
| [Jomarkow Hard Hat Workers](https://roboflow.com) | 1,000 | YOLO TXT | PPE + person detection training |
| [INRIA Person Dataset](https://huggingface.co/datasets/marcelarosalesj/inria-person) | 300 crops | HuggingFace | Person detection training |

**Combined training set:** 3,626 crops (80/20 train/val split)

---

## Model Results

### Multi-class Classification (5 classes)

| Model | Accuracy | Macro F1 | AUC |
|-------|----------|----------|-----|
| CNN PPENet (100 epochs, GPU) | **87.33%** | **0.856** | **0.978** |
| Ensemble (SVM+RF+ET+GBM) | 79.48% | 0.777 | 0.950 |
| HistGBM (400 rounds, no PCA) | 76.72% | 0.745 | 0.943 |
| SVM (PCA 220 → RBF, balanced) | 76.31% | 0.745 | 0.943 |
| Random Forest (400 trees) | 73.00% | 0.708 | 0.923 |
| ExtraTrees (400 trees) | 71.76% | 0.693 | 0.924 |

### Per-class CNN F1 Scores (100 epochs)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| safety_vest | 0.93 | 0.93 | **0.93** |
| no_ppe | 0.87 | 0.94 | **0.90** |
| helmet | 0.92 | 0.91 | 0.91 |
| partial_ppe | 0.75 | 0.77 | 0.76 |
| full_ppe | 0.85 | 0.71 | 0.77 |

### Person Detection (YOLOv8n fine-tuning)

| Epoch | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| 1 | 0.288 | 0.446 | 0.313 |
| 2 | 0.443 | 0.506 | 0.489 |
| 3 | 0.608 | 0.660 | 0.590 |
| 4 | **0.679** | **0.708** | **0.613** |

---

## Repository Structure

```
PPE-Detection/
├── src/
│   ├── ppe_production_train.py    # Full production training script (all models)
│   ├── ppe_cnn_fast.py            # PPENetFast CNN definition
│   ├── ppe_ml_models.py           # SVM, RF, ExtraTrees, GBM baselines
│   ├── ppe_combined_pipeline.py   # YOLO person detect + PPENetFast classify
│   ├── ppe_cctv_validation.py     # CCTV sliding-window validation
│   └── ppe_pipeline.py            # Single-image inference pipeline
├── person_detection/
│   ├── person_detect.yaml         # YOLOv8 training config
│   └── prepare_dataset.py         # Head-to-body box expansion script
├── reports/
│   ├── create_prod_report.js      # Word document generator (v3)
│   └── create_report.js           # Word document generator (v1)
├── skills/
│   └── ppe-report-generator/      # Claude Code skill: HTML evaluation report
├── results/
│   ├── plots/                     # Evaluation visualisations
│   ├── models/                    # Saved weights (.pth, .pkl) and plots
│   └── reports/                   # HTML evaluation report + ROC curves
├── setup_datasets.py              # Dataset download and preparation
└── docs/
    └── PPE_Detection_Report_v3.docx
```

---

## How to Run

### Install dependencies
```bash
pip install torch torchvision scikit-learn opencv-python ultralytics matplotlib seaborn pandas joblib
```

### Setup datasets
```bash
python setup_datasets.py
```

### Train production models (CNN + all ML baselines)
```bash
python src/ppe_production_train.py --epochs 100 --batch-size 256 --max-per-class 600
```

### Run two-stage pipeline on an image
```bash
python src/ppe_combined_pipeline.py
```

### Fine-tune YOLOv8n person detector
```bash
yolo train data=person_detection/person_detect.yaml model=yolov8n.pt epochs=10 imgsz=416 freeze=9
```

### Run CCTV validation
```bash
python src/ppe_cctv_validation.py
```

### Generate Word document report
```bash
cd reports && node create_prod_report.js
```

---

## Ethical Considerations

- **Face anonymisation** should be applied to all detected person crops before storage
- **Data retention:** 24–72 hours for non-incident footage recommended
- **Worker notification** required via signage and policy documentation
- Model trained primarily on outdoor construction scenes — may underperform in some industrial settings
- Tune `no_ppe` confidence threshold lower (0.35) to favour safety recall over precision

---

## Future Improvements

1. **More YOLOv8n fine-tuning** — 4 epochs trained (mAP50=0.679), target 0.85+ with 20+ epochs
2. **Transfer learning** (MobileNetV2 / EfficientNet-B0) — expected +5–8%
3. **Data augmentation** during training — expected +3–5%
4. **YOLO end-to-end** (detect + classify in one pass) — expected +10–15%
5. **More `partial_ppe` / `full_ppe` training data** — addresses weakest classes (F1 0.76/0.77)
