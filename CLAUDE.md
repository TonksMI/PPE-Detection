# PPE Detection Project — Claude Code Context

## Project Goal
Two-stage pipeline: (1) detect people in images, (2) classify PPE on each person.
Dataset: Jomarkow (safety site images, head annotations) + MinhNKB (labeled PPE crops).
5 classes: helmet, safety_vest, no_ppe, partial_ppe, full_ppe

## Repo Structure
PPE-Detection/
├── setup_datasets.py           # dataset download/preparation script
├── src/                        # all training scripts
│   ├── ppe_ml_models.py        # SVM / RF / GBM (HOG features)
│   ├── ppe_cnn.py              # CNN v1 (PPENet, 64×64)
│   ├── ppe_cnn_fast.py         # PPENetFast (32×32, production model)
│   ├── ppe_cctv_validation.py  # sliding-window HOG+CNN CCTV test
│   ├── ppe_pipeline.py         # single-image inference pipeline
│   ├── ppe_combined_pipeline.py # YOLO person detect + PPENetFast classify
│   ├── ppe_production_train.py # full pipeline: loads cache, trains all models
│   ├── ppe_ml_continue.py      # ML model training continuation
│   ├── ppe_v2_cnn_only.py      # CNN v2 experiments
│   └── ppe_final_report.py     # evaluation plots + report generation
├── person_detection/
│   ├── prepare_dataset.py      # builds YOLO training set (head→body expansion + INRIA crops)
│   └── person_detect.yaml      # YOLOv8n config (nc=1, person)
├── reports/
│   ├── create_prod_report.js   # generates PPE_Detection_Report_v2.docx via docx@9.5.1
│   └── create_report.js        # v1 report generator
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
- **Platform**: Windows 10/11 (win32), Python 3.10.6 (MSC v.1932 64-bit AMD64)
- **PyTorch**: 2.12.0.dev (nightly) + CUDA 12.8 + torchvision 0.26.0.dev
- **GPU**: NVIDIA GeForce RTX 5070 (12GB VRAM, Blackwell sm_120)
- **Working directory**: D:\Claude\PPE-Detection
- **Datasets dir**: D:\datasets\ (jomarkow, helmet-safety-vest-detection-master, inria_person, ppe_crops, person_detection)
- Cache dir: D:\Claude\cache\
- All final outputs go in results/plots/, results/summaries/, results/models/
- Commit changes with descriptive messages following existing style (feat:, results:, fix:, docs:)

### Currently Installed
- torch 2.12.0.dev+cu128, torchvision 0.26.0.dev+cu128
- ultralytics 8.4.17, scikit-learn 1.7.2, opencv-python 4.13.0.92
- joblib 1.5.3, matplotlib 3.10.8, numpy 2.2.6, pandas 2.3.3, pillow 12.1.0

### Training Notes (CUDA)
- GPU training enabled — expect 5-10x speedup vs CPU
- RTX 5070 has 12GB VRAM — can use batch=256 or higher for PPENet
- YOLOv8n fine-tuning: use `device=0` or `device='cuda'`
- Mixed precision (fp16) supported for faster training

## Priorities for Improvement
1. More YOLOv8n fine-tuning epochs (only 4 done, mAP50=0.679; target 0.85+)
2. Transfer learning for PPE classifier (MobileNetV2 or EfficientNet-B0)
3. Data augmentation (currently minimal)
4. Class weighting for partial_ppe / full_ppe (underrepresented)
5. End-to-end YOLO for PPE detection (single-stage)

## Commands

```bash
# Train PPENetFast (production CNN)
python src/ppe_production_train.py

# Train ML baselines (SVM/RF/GBM)
python src/ppe_ml_models.py

# Run full two-stage pipeline (YOLO + CNN) on an image
python src/ppe_combined_pipeline.py

# CCTV sliding-window validation
python src/ppe_cctv_validation.py

# Setup/download datasets
python setup_datasets.py
```

## Conventions
- Always save new scripts to src/, new plots to results/plots/, updated models to results/models/
- Commit after completing each meaningful unit of work
