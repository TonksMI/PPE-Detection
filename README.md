# PPE Detection — Computer Vision Pipeline

Automated Personal Protective Equipment detection for industrial environments. The project spans three assignments, each extending the previous one: crop-level classification → object detection → pixel-level segmentation.

---

## Project Arc

| Assignment | Task | Best Result |
|---|---|---|
| 1 — Classification | 5-class crop classification (MinhNKB schema) | ViT-B/16: **93.90%** |
| 2 — Object Detection | End-to-end 5-class detection on full scenes | YOLOv8n: **86.3% mAP50** |
| 3 — Segmentation | Pixel-wise & instance segmentation (keremberke 10-class) | YOLOv8n-seg: **90.0% box / 87.1% mask mAP50** |

---

## Schemas

Two separate label schemas are used across the project. They come from different datasets and are **not interchangeable**.

### MinhNKB 5-class (Assignments 1 & 2)

Whole-person crop labels — one label per person bounding box.

| Class | Meaning |
|---|---|
| `helmet` | Person wearing a hard hat only (no vest) |
| `safety_vest` | Person wearing a hi-vis vest only (no hard hat) |
| `full_ppe` | Person wearing both hard hat and hi-vis vest |
| `partial_ppe` | Person wearing some but not all required PPE |
| `no_ppe` | Person wearing neither hard hat nor vest |

No safety glasses, gloves, or footwear in this schema.

### Keremberke 10-class (Assignment 3)

Individual item labels — one label per PPE item bounding box, not per person.

| Class | Meaning |
|---|---|
| `helmet` / `no_helmet` | Hard hat present / absent on the head |
| `glove` / `no_glove` | Safety glove present / absent on the hand |
| `goggles` / `no_goggles` | Protective eyewear present / absent |
| `mask` / `no_mask` | Face mask present / absent |
| `shoes` / `no_shoes` | Safety footwear present / absent |

No safety vests, full_ppe, or partial_ppe in this schema.

---

## All Model Results

### Assignment 1 — Classification (MinhNKB 5-class, 80/20 split)

| Model | Val Accuracy | Notes |
|---|---|---|
| ViT-B/16 (fine-tuned) | **93.90%** | Best overall |
| Custom PPENet CNN | 87.33% | 3 conv blocks, 226K params |
| SAM2-masked CNN | 74.48% | Background removal hurt accuracy |
| ResNet-18 (transfer, frozen) | ~79.82% | Only final layer trained |
| ResNet-18 (scratch) | ~79.23% | No pretrained weights |
| Random Forest (400 trees) | 79.48% | HOG + colour features |
| SVM (RBF, PCA-220) | 76.31% | Best classical model |

### Assignment 2 — Object Detection (MinhNKB 5-class, 2609 scenes)

| Model | mAP50 | Notes |
|---|---|---|
| YOLOv8n end-to-end | **86.3%** | Single-stage, full scene |

### Assignment 3 — Segmentation (keremberke 10-class, 4000 images)

| Model | Task | Box mAP50 | Mask mAP50 | mIoU | Pixel Acc |
|---|---|---|---|---|---|
| YOLOv8n-seg | Instance seg (10-class) | **90.0%** | **87.1%** | — | — |
| DeepLabV3+ ResNet50 | Semantic seg (11-class) | — | — | pending | pending |

#### YOLOv8n-seg per-class (keremberke val set, 600 images)

| Class | Box mAP50 | Mask mAP50 | Ann. count |
|---|---|---|---|
| helmet | 99.3% | 99.1% | 1,523 |
| no_helmet | 91.9% | 88.7% | 1,296 |
| glove | 87.6% | 86.7% | 4,663 |
| no_glove | 80.8% | 75.6% | 6,126 |
| goggles | 87.1% | 79.6% | 4,184 |
| no_goggles | 81.2% | 76.7% | 4,092 |
| mask | 96.1% | 91.9% | 269 |
| no_mask | 82.4% | 78.8% | 661 |
| shoes | 94.5% | 94.5% | 755 |
| no_shoes | 99.5% | 99.5% | 606 |

---

## Repository Structure

```
PPE-Detection/
│
├── src/                                  # All Python training and inference scripts
│   │
│   ├── ── Core Pipeline (Assignment 1 & 2) ──
│   ├── ppe_production_train.py           # Train CNN + SVM/RF/ET/GBM ensemble
│   ├── ppe_cnn_fast.py                   # PPENetFast architecture (226K params)
│   ├── ppe_ml_models.py                  # Classical ML baselines
│   ├── ppe_combined_pipeline.py          # YOLO person detect → PPENet classify
│   ├── ppe_pipeline.py                   # Single-image inference helper
│   ├── ppe_cctv_validation.py            # CCTV sliding-window validation
│   ├── ppe_early_stopping.py             # EarlyStopping callback utility
│   │
│   ├── ── Transfer Learning ──
│   ├── ppe_vit_train.py                  # ViT-B/16 fine-tuning (93.90%)
│   ├── ppe_vit_evaluate.py               # ViT evaluation + confusion matrix
│   ├── ablation_study.py                 # Frozen vs fine-tuned ablation
│   │
│   ├── ── Assignment 3 — Segmentation ──
│   ├── ppe_seg_keremberke_rebuild.py     # Build 10-class dataset from zips (SAM2)
│   ├── ppe_seg_data_prep.py              # MinhNKB SAM2 data prep (5-class, legacy)
│   ├── ppe_seg_keremberke_local.py       # Supplement keremberke to MinhNKB (legacy)
│   ├── ppe_deeplab_train.py              # DeepLabV3+ ResNet50 semantic segmentation
│   ├── ppe_yolo_seg_train.py             # YOLOv8n-seg instance segmentation
│   ├── _gen_deeplab_predgrid.py          # Regenerate DeepLab prediction grid plot
│   │
│   ├── ── Experiments ──
│   ├── ppe_masked_cnn_train.py           # SAM2-masked CNN ablation (74.48%)
│   ├── ppe_mask_generator.py             # SAM2 binary mask generation utility
│   ├── ppe_rnn_train.py                  # RNN/LSTM/GRU experiments
│   ├── ppe_unet_train.py                 # UNet segmentation experiment
│   │
│   ├── ── Reporting & Visualisation ──
│   ├── ppe_experiment_comparison.py      # 19-model comparison chart
│   ├── ppe_full_evaluation_report.py     # ROC curves, confusion matrices, PR curves
│   ├── ppe_final_report.py               # Summary report generator
│   ├── ppe_cctv_report.py                # CCTV validation report
│   ├── replot_individual_heatmaps.py     # Re-render individual heatmap plots
│   │
│   └── ── Legacy Iterations ──
│       ├── ppe_cnn.py                    # Original CNN (superseded by ppe_cnn_fast.py)
│       ├── ppe_v2_cnn_only.py            # v2 CNN-only iteration
│       ├── ppe_ml_continue.py            # Ad-hoc continued ML training
│       └── setup_yolo_ppe.py             # Old YOLO dataset setup
│
├── person_detection/                     # Person detector fine-tuning
│   ├── person_detect.yaml                # YOLOv8 training config
│   ├── prepare_dataset.py                # Head-to-body bounding box expansion
│   └── train_yolo_end2end.py             # End-to-end YOLOv8 PPE training
│
├── reports/                              # Document generators
│   ├── generate_assignment_report.py     # Final Project Writeup (Python/python-docx)
│   ├── gen_assignment3.js                # Assignment 3 Writeup (Node.js/docx)
│   ├── create_prod_report.js             # Production report (Node.js, v3)
│   ├── create_report.js                  # Original report (Node.js, v1)
│   └── package.json                      # Node.js dependencies (docx)
│
├── results/
│   ├── models/                           # Saved weights + per-model result plots
│   │   ├── prod_cnn_model.pth            # PPENetFast best weights
│   │   ├── yolo_e2e_best.pt              # YOLOv8n end-to-end best weights
│   │   ├── yolo_seg_best.pt              # YOLOv8n-seg best weights
│   │   ├── person_detector_yolov8n.pt    # Fine-tuned person detector
│   │   ├── *_results.csv                 # Per-model result CSVs
│   │   └── *.png                         # Confusion matrices, training curves
│   ├── plots/                            # EDA and training visualisations
│   ├── reports/                          # HTML evaluation report
│   └── summaries/                        # Aggregate CSVs and ablation results
│
├── docs/                                 # Generated write-up documents
│   ├── Final_Project_Writeup.docx        # Full project report (python-docx)
│   ├── Assignment3_Writeup.docx          # Assignment 3 segmentation report
│   ├── PPE_Detection_Report_v3.docx      # Earlier production report
│   └── PPE_Detection_Report_v2.docx      # Earlier draft
│
├── cctv_validation_original/             # CCTV test images for pipeline validation
├── logs/                                 # Training log files (gitignored)
├── runs/                                 # YOLO training artefacts (gitignored)
├── sam2_b.pt                             # SAM2-B checkpoint (used by seg scripts)
├── yolov8n.pt                            # YOLOv8n base weights
├── setup_datasets.py                     # Dataset download and preparation
└── CLAUDE.md                             # Claude Code project instructions
```

---

## Datasets

| Dataset | Images | Format | Used for |
|---|---|---|---|
| [MinhNKB Helmet-Safety-Vest](https://github.com/MinhNKB/helmet-safety-vest-detection) | 1,613 | Pascal VOC XML | Classification + semantic seg (5-class) |
| [Jomarkow Hard Hat Workers](https://roboflow.com) | ~1,000 | YOLO TXT | End-to-end detection training |
| [keremberke/protective-equipment-detection](https://huggingface.co/datasets/keremberke/protective-equipment-detection) | ~12,000 | COCO JSON | Segmentation (10-class, all 3 zips) |
| [INRIA Person Dataset](https://huggingface.co/datasets/marcelarosalesj/inria-person) | 300 crops | HuggingFace | Person detector training |

Datasets are stored at `D:/Claude/datasets/` (not in repo). The keremberke zips (`train.zip`, `test.zip`, `valid.zip`) must be placed in the project root before running `ppe_seg_keremberke_rebuild.py`.

---

## How to Run

### Install dependencies
```bash
pip install torch torchvision scikit-learn opencv-python ultralytics matplotlib seaborn pandas joblib python-docx
npm install   # inside reports/ for document generation
```

### Setup MinhNKB + jomarkow datasets
```bash
python setup_datasets.py
```

### Train production CNN + ML ensemble (Assignment 1)
```bash
python src/ppe_production_train.py --epochs 100 --batch-size 256 --max-per-class 600
```

### Fine-tune ViT-B/16 (Assignment 1, best classifier)
```bash
python src/ppe_vit_train.py
```

### Run two-stage inference pipeline
```bash
python src/ppe_combined_pipeline.py
```

### Train YOLOv8n end-to-end detector (Assignment 2)
```bash
python person_detection/train_yolo_end2end.py
```

### Build keremberke segmentation dataset (Assignment 3)
```bash
# Place train.zip, test.zip, valid.zip in project root first
python src/ppe_seg_keremberke_rebuild.py
```

### Train DeepLabV3+ semantic segmentation (Assignment 3)
```bash
python src/ppe_deeplab_train.py
```

### Train YOLOv8n-seg instance segmentation (Assignment 3)
```bash
python src/ppe_yolo_seg_train.py
```

### Generate comparison chart (all 19+ models)
```bash
python src/ppe_experiment_comparison.py
```

### Generate Word document reports
```bash
# Final project report
python reports/generate_assignment_report.py

# Assignment 3 report
cd reports && node gen_assignment3.js
```

---

## Environment

| | |
|---|---|
| Python | 3.10.6 |
| PyTorch | 2.12.0.dev+cu128 |
| CUDA | 12.8 |
| GPU | RTX 5070 (12 GB VRAM) |
| OS | Windows 11 |

---

## Ethical Considerations

- **Face anonymisation** should be applied to all person crops before storage or logging.
- **Data retention:** 24–72 hours recommended for non-incident footage.
- **Worker notification** is required — signage and written policy.
- Models trained on construction and warehouse scenes may underperform in other industrial environments.
- The keremberke 10-class model should not be used for whole-person PPE compliance decisions — it detects individual items, not overall compliance state.
- Tune `no_ppe` / `no_helmet` confidence thresholds lower (≈0.35) to favour safety recall over precision.
