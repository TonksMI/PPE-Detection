# PPE Detection Project — Claude Code Context

## Project Goal
Two-stage pipeline: (1) detect people in images, (2) classify PPE on each person.
Dataset: Jomarkow (safety site images, head annotations) + MinhNKB (labeled PPE crops).
5 classes: helmet, safety_vest, no_ppe, partial_ppe, full_ppe

## Repo Structure
PPE-Detection/
├── src/                        # all training scripts
│   ├── eda_analysis.py         # EDA + feature extraction
│   ├── train_models.py         # SVM / RF / GBM (HOG features)
│   ├── train_cnn.py            # CNN v1 (PPENet, 64×64)
│   ├── cctv_validation.py      # sliding-window HOG+CNN CCTV test
│   ├── ppe_production_train.py # full pipeline: loads cache, trains all models
│   └── generate_plots.py       # all evaluation plots
├── person_detection/
│   ├── prepare_dataset.py      # builds YOLO training set (head→body expansion + INRIA crops)
│   └── person_detect.yaml      # YOLOv8n config (nc=1, person)
├── reports/
│   └── create_prod_report.js   # generates PPE_Detection_Report_v2.docx via docx@9.5.1
├── results/
│   ├── plots/                  # all .png evaluation figures
│   ├── summaries/              # CSV performance summaries
│   └── models/                 # prod_cnn_model.pth (PPENetFast), person_detector_yolov8n.pt
└── docs/
    └── PPE_Detection_Report_v2.docx

## Key Technical Decisions
- CNN: PPENetFast (32×32 input, 3 conv blocks 32→64→128, AdaptiveAvgPool(2,2), FC 512→256→5, 226K params)
  - Trained 30 epochs, batch=256, OneCycleLR max_lr=1e-3, label_smoothing=0.05 → 87.05% val acc
- ML models: SVM (PCA 150 + rbf, C=15) 76.6%, RF (400 trees) 71.8%, GBM 67.2%
- Person detection: fine-tuned YOLOv8n (ultralytics) replacing OpenCV HOG
  - Fine-tuned 4 epochs on jomarkow head→body expansion + 300 INRIA crops → mAP50=0.679
  - Weights: results/models/person_detector_yolov8n.pt
- Disk cache: crops_X_600.npy, crops_y_600.npy, features_600.npy (3,626 crops, 32×32)

## Current Model Results
- PPENetFast CNN: helmet F1=0.94, vest F1=0.95, no_ppe F1=0.89, partial_ppe F1=0.72, full_ppe F1=0.77
- Binary (ppe/no_ppe): SVM 84.6%, RF 82.1%, GBM 81.8%
- YOLOv8n zero-shot: 36 people across 14 CCTV images @ 83.2% avg confidence

## Environment
- Python 3.x, PyTorch (CPU only), ultralytics, scikit-learn, opencv-python, joblib, matplotlib
- Cache dir: ../cache/ (relative to src/)
- All final outputs go in results/plots/, results/summaries/, results/models/
- Commit changes with descriptive messages following existing style (feat:, results:, fix:, docs:)

## Priorities for Improvement
1. More YOLOv8n fine-tuning epochs (only 4 done, mAP50=0.679; target 0.85+)
2. Transfer learning for PPE classifier (MobileNetV2 or EfficientNet-B0)
3. Data augmentation (currently minimal)
4. Class weighting for partial_ppe / full_ppe (underrepresented)
5. End-to-end YOLO for PPE detection (single-stage)

## Conventions
- Always save new scripts to src/, new plots to results/plots/, updated models to results/models/
- Commit after completing each meaningful unit of work
