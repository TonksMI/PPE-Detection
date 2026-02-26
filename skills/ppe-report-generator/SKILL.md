---
name: ppe-report-generator
description: >
  Generates a comprehensive evaluation report for the PPE Detection pipeline.
  Use this skill whenever the user asks to generate a report, view results,
  see ROC/AUC curves, evaluate model performance, summarize validation,
  show plots, or check how the PPE models are doing. Triggers on phrases
  like "generate report", "show me the results", "ROC curves", "AUC",
  "validation report", "model evaluation", "how are the models doing",
  "pipeline report", or any request for a performance summary of the PPE system.
---

# PPE Pipeline Report Generator

Generates a full HTML evaluation report covering model performance, ROC/AUC curves
for every model, training history, confusion matrices, and CCTV validation results.

## What gets produced

- **Multi-class ROC curves** (one-vs-rest) for every model — SVM, RF, ExtraTrees, GBM, Ensemble, CNN
- **Binary ROC curves** for all binary classifiers
- **Precision-Recall curves** per model and per class
- **Training history** (CNN loss/accuracy over epochs, if available)
- **Confusion matrices** for all multi-class models
- **Per-class F1 heatmap** across all models
- **CCTV validation summary** (people detected, classified, violations per image)
- **Model comparison** bar chart (accuracy + macro F1)
- **Self-contained HTML report** at `results/reports/ppe_evaluation_report.html`

## Workflow

### Step 1 — Check prerequisites

Verify these files exist before proceeding:
- `D:/Claude/cache/features_600.npy` — HOG+color feature matrix
- `D:/Claude/cache/crops_y_600.npy` — class labels
- `results/models/prod_model_summary.csv` — performance summary
- At least one of the `results/models/prod_*.pkl` model files

If cache files are missing, tell the user to run `python src/ppe_production_train.py` first.

### Step 2 — Run the bundled script

The report script lives at `skills/ppe-report-generator/scripts/generate_report.py`.
Run it from the project root:

```bash
cd D:/Claude/PPE-Detection
PYTHONIOENCODING=utf-8 python skills/ppe-report-generator/scripts/generate_report.py
```

The script:
- Loads cached features + labels, splits 80/20 (random_state=42) matching training
- Loads every `prod_*.pkl` model that exists in `results/models/`
- Computes one-vs-rest ROC/AUC for multi-class models (5 classes)
- Computes standard ROC/AUC for binary models
- Loads the CNN checkpoint and computes CNN ROC curves on the same test split
- Assembles all plots and the model summary into a self-contained HTML file

### Step 3 — Report to the user

After the script completes, tell the user:
- Where the HTML report was saved (`results/reports/ppe_evaluation_report.html`)
- The top-line AUC scores for each model (printed by the script)
- Any warnings (e.g., missing models that were skipped)

## Output location

All outputs go to `results/reports/`:
- `ppe_evaluation_report.html` — the main self-contained report (all plots embedded as base64)
- `roc_multiclass.png` — multi-class one-vs-rest ROC grid
- `roc_binary.png` — binary ROC curves
- `pr_curves.png` — precision-recall curves

## Notes

- The script reuses the **same 80/20 test split** (random_state=42, stratified) used during
  training, so AUC numbers are directly comparable to the accuracy figures in the summary CSV.
- CNN ROC curves require `results/models/prod_cnn_model.pth` and
  `results/models/prod_le_cnn.pkl`. If missing, the CNN section is skipped gracefully.
- The ensemble `.pkl` is large (~128 MB); loading it takes a few seconds.
- All plots are embedded as base64 in the HTML so the report is fully portable.
