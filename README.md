# PPE Detection — Computer Vision Final Project

Automated Personal Protective Equipment (PPE) detection for industrial warehouse environments using a two-stage computer vision pipeline.

## Overview

| Stage | Method | Accuracy |
|-------|--------|----------|
| Person Detection | YOLOv8n fine-tuned (mAP50 = 0.679) | Replaces OpenCV HOG |
| PPE Classification | CNN PPENet — 30 epochs | **87.1% multi-class** |
| Binary (PPE / no PPE) | SVM (HOG + colour features) | **84.6%** |

### Pipeline

```
CCTV Frame → YOLOv8n Person Detector → Person Crops → CNN PPENet → PPE Classification
```

**Stage 1 — Person Detection:** YOLOv8n fine-tuned on 1,105 images (jomarkow hard-hat workers with head-to-body box expansion + 300 INRIA pedestrian crops). Pre-trained on COCO (118K images).

**Stage 2 — PPE Classification:** Custom lightweight CNN (226K parameters, 32×32 input) classifying 5 safety classes with 87.1% validation accuracy.

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

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| CNN PPENet (30 epochs) | **87.05%** | **0.849** |
| SVM (PCA → RBF) | 76.58% | 0.749 |
| Random Forest (400 trees) | 71.76% | 0.692 |
| HistGBM (200 rounds) | 67.22% | 0.642 |

### Per-class CNN F1 Scores

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| helmet | 0.93 | 0.95 | **0.94** |
| safety_vest | 0.94 | 0.97 | **0.95** |
| no_ppe | 0.90 | 0.88 | 0.89 |
| full_ppe | 0.79 | 0.74 | 0.77 |
| partial_ppe | 0.72 | 0.72 | 0.72 |

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
│   ├── ppe_pipeline.py            # Original end-to-end pipeline
│   ├── ppe_ml_models.py           # SVM, RF, GBM training (v1)
│   ├── ppe_ml_continue.py         # ML models with combined dataset
│   ├── ppe_cnn_fast.py            # CNN v1 (pre-cached crops)
│   ├── ppe_production_train.py    # Full production training script
│   ├── ppe_cctv_validation.py     # CCTV sliding-window validation
│   └── ppe_final_report.py        # Dashboard and visualisation
├── person_detection/
│   ├── person_detect.yaml         # YOLOv8 training config
│   └── prepare_dataset.py         # Head-to-body box expansion script
├── reports/
│   ├── create_report.js           # Word document generator (v1)
│   └── create_prod_report.js      # Word document generator (v2)
├── results/
│   ├── plots/                     # All evaluation visualisations
│   ├── summaries/                 # CSV model comparison tables
│   └── models/                    # Saved model weights (.pth)
└── docs/
    └── PPE_Detection_Report_v2.docx
```

---

## How to Run

### Install dependencies
```bash
pip install torch torchvision scikit-learn opencv-python ultralytics matplotlib seaborn pandas joblib
```

### Train production models
```bash
python src/ppe_production_train.py --epochs 30 --batch-size 256 --max-per-class 600
```

### Fine-tune YOLOv8n person detector
```bash
yolo train data=person_detection/person_detect.yaml model=yolov8n.pt epochs=10 imgsz=416 freeze=9
```

### Run CCTV validation
```bash
python src/ppe_cctv_validation.py
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

1. **Transfer learning** (MobileNetV2 / EfficientNet-B0) — expected +5–8%
2. **Data augmentation** during training — expected +3–5%
3. **YOLO end-to-end** (detect + classify in one pass) — expected +10–15%
4. **More `partial_ppe` / `full_ppe` training data** — addresses weakest classes
5. **GPU training** — enables 100+ epochs and full 64×64 crops
