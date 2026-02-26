"""
PPE Detection — Full Pipeline Evaluation Report
================================================
Generates a comprehensive evaluation report with:
  - ROC & AUC curves for all binary and multi-class models (OvR)
  - Precision-Recall curves
  - Per-class F1 heatmap
  - Confusion matrices for all models
  - Model comparison bar charts
  - Summary statistics dashboard

Models covered:
  ML (binary):  SVM, RandomForest, ExtraTrees, HistGBM
  ML (multi):   SVM, RandomForest, ExtraTrees, HistGBM, Ensemble
  CNN:          PPENet (64x64)
"""

import os, sys, pickle, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve, average_precision_score,
    accuracy_score
)

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CACHE_DIR   = "D:/Claude/cache"
MODELS_DIR  = os.path.join(PROJECT_DIR, "results", "models")
OUT_DIR     = os.path.join(PROJECT_DIR, "results", "models")   # same dir per convention

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Colour palette ─────────────────────────────────────────────────────────────
CLASS_COLORS = {
    'full_ppe':    '#8e44ad',
    'helmet':      '#27ae60',
    'no_ppe':      '#e74c3c',
    'partial_ppe': '#e67e22',
    'safety_vest': '#2980b9',
}

MODEL_COLORS = {
    'SVM':        '#e74c3c',
    'RF':         '#3498db',
    'ExtraTrees': '#2ecc71',
    'HistGBM':    '#f39c12',
    'Ensemble':   '#9b59b6',
    'CNN':        '#1abc9c',
}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Loading cached data ...")
MAX_CLASS = 600
crops_rgb  = np.load(os.path.join(CACHE_DIR, f"crops_X_{MAX_CLASS}.npy"))
labels_raw = np.load(os.path.join(CACHE_DIR, f"crops_y_{MAX_CLASS}.npy"))
X_ml       = np.load(os.path.join(CACHE_DIR, f"features_{MAX_CLASS}.npy"))

print(f"  Crops  : {crops_rgb.shape}")
print(f"  Labels : {labels_raw.shape}  unique={np.unique(labels_raw)}")
print(f"  Features: {X_ml.shape}")

# Load label encoders
with open(os.path.join(MODELS_DIR, "prod_le_multi.pkl"),  "rb") as f: le_multi  = pickle.load(f)
with open(os.path.join(MODELS_DIR, "prod_le_binary.pkl"), "rb") as f: le_binary = pickle.load(f)
with open(os.path.join(MODELS_DIR, "prod_le_cnn.pkl"),    "rb") as f: le_cnn    = pickle.load(f)

ALL_CLASSES    = list(le_multi.classes_)   # alphabetical
BINARY_CLASSES = list(le_binary.classes_)
N_CLASSES = len(ALL_CLASSES)

print(f"  Multi  classes: {ALL_CLASSES}")
print(f"  Binary classes: {BINARY_CLASSES}")

# Encode labels
BINARY_MAP = {c: ("ppe_present" if c != "no_ppe" else "no_ppe") for c in ALL_CLASSES}
y_multi  = le_multi.fit_transform(labels_raw)
y_binary = le_binary.fit_transform([BINARY_MAP[l] for l in labels_raw])

# Reproducible split (same seed as training script)
Xtr, Xte, ytr, yte         = train_test_split(X_ml, y_multi,  test_size=0.2, random_state=42, stratify=y_multi)
Xtrb, Xteb, ytrb, yteb     = train_test_split(X_ml, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# CNN image split (same seed)
tr_idx, te_idx = train_test_split(range(len(y_multi)), test_size=0.2, random_state=42, stratify=y_multi)
y_cnn    = le_cnn.transform(labels_raw)
yte_cnn  = y_cnn[te_idx]

print(f"  Multi  split: train={len(Xtr)} test={len(Xte)}")
print(f"  Binary split: train={len(Xtrb)} test={len(Xteb)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOAD TRAINED ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Loading trained ML models ...")

ml_multi = {
    'SVM':        joblib.load(os.path.join(MODELS_DIR, "prod_svm_multi.pkl")),
    'RF':         joblib.load(os.path.join(MODELS_DIR, "prod_rf_multi.pkl")),
    'ExtraTrees': joblib.load(os.path.join(MODELS_DIR, "prod_et_multi.pkl")),
    'HistGBM':    joblib.load(os.path.join(MODELS_DIR, "prod_gbm_multi.pkl")),
    'Ensemble':   joblib.load(os.path.join(MODELS_DIR, "prod_ensemble_multi.pkl")),
}
ml_binary = {
    'SVM':        joblib.load(os.path.join(MODELS_DIR, "prod_svm_binary.pkl")),
    'RF':         joblib.load(os.path.join(MODELS_DIR, "prod_rf_binary.pkl")),
    'ExtraTrees': joblib.load(os.path.join(MODELS_DIR, "prod_et_binary.pkl")),
    'HistGBM':    joblib.load(os.path.join(MODELS_DIR, "prod_gbm_binary.pkl")),
}
print(f"  Multi models : {list(ml_multi.keys())}")
print(f"  Binary models: {list(ml_binary.keys())}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  LOAD CNN + RUN INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Loading CNN and running inference ...")

class PPENet(nn.Module):
    """PPENet as defined in ppe_production_train.py (64x64 input)"""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

ckpt = torch.load(os.path.join(MODELS_DIR, "prod_cnn_model.pth"),
                  map_location=DEVICE, weights_only=False)
cnn_model = PPENet(len(le_cnn.classes_)).to(DEVICE)
cnn_model.load_state_dict(ckpt['state_dict'])
cnn_model.eval()
print(f"  CNN loaded — arch={ckpt['arch']}  epoch={ckpt['epoch']}  best_val={ckpt['best_val_acc']:.4f}")

# Prepare test crops tensor (normalise as in training)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
crops_norm = (crops_rgb.astype(np.float32) / 255.0 - mean) / std      # (N,64,64,3)
X_cnn_t = torch.tensor(crops_norm.transpose(0, 3, 1, 2))              # (N,3,64,64)
X_cnn_te = X_cnn_t[te_idx]

BATCH = 256
all_probs_cnn = []
with torch.no_grad():
    for i in range(0, len(X_cnn_te), BATCH):
        batch = X_cnn_te[i:i+BATCH].to(DEVICE)
        logits = cnn_model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs_cnn.append(probs)

probs_cnn = np.vstack(all_probs_cnn)                  # (N_test, 5)
preds_cnn = probs_cnn.argmax(axis=1)

cnn_acc = accuracy_score(yte_cnn, preds_cnn)
cnn_rpt = classification_report(yte_cnn, preds_cnn,
                                  target_names=le_cnn.classes_, output_dict=True)
cnn_cm  = confusion_matrix(yte_cnn, preds_cnn)
print(f"  CNN test accuracy: {cnn_acc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  COLLECT PREDICTIONS & PROBABILITIES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Collecting predictions ...")

multi_results  = {}
binary_results = {}

for name, model in ml_multi.items():
    yp    = model.predict(Xte)
    proba = model.predict_proba(Xte)
    rpt   = classification_report(yte, yp, target_names=le_multi.classes_, output_dict=True)
    cm    = confusion_matrix(yte, yp)
    multi_results[name] = {
        'yp': yp, 'proba': proba, 'report': rpt, 'cm': cm,
        'accuracy': rpt['accuracy']
    }
    print(f"  {name:15s} multi  acc={rpt['accuracy']:.4f}")

for name, model in ml_binary.items():
    yp    = model.predict(Xteb)
    proba = model.predict_proba(Xteb)
    rpt   = classification_report(yteb, yp, target_names=le_binary.classes_, output_dict=True)
    cm    = confusion_matrix(yteb, yp)
    binary_results[name] = {
        'yp': yp, 'proba': proba, 'report': rpt, 'cm': cm,
        'accuracy': rpt['accuracy']
    }
    print(f"  {name:15s} binary acc={rpt['accuracy']:.4f}")

# Add CNN to multi results
multi_results['CNN'] = {
    'yp': preds_cnn, 'proba': probs_cnn, 'report': cnn_rpt, 'cm': cnn_cm,
    'accuracy': cnn_acc
}
print(f"  {'CNN':15s} multi  acc={cnn_acc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PLOT 1 — BINARY ROC & AUC CURVES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Generating plots ...")

# --- 5a. Binary ROC curves ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("ROC Curves — Binary PPE Classification (PPE Present vs No PPE)",
             fontsize=14, fontweight='bold')

# Left: individual binary model curves
ax = axes[0]
ppe_present_idx = list(le_binary.classes_).index('ppe_present')

for name, res in binary_results.items():
    proba_pos = res['proba'][:, ppe_present_idx]
    fpr, tpr, _ = roc_curve(yteb, proba_pos, pos_label=ppe_present_idx)
    roc_auc     = auc(fpr, tpr)
    col         = MODEL_COLORS.get(name, 'gray')
    ax.plot(fpr, tpr, lw=2.5, color=col,
            label=f"{name}  (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random (AUC = 0.500)')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Binary ML Models — ROC", fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

# Right: zoom into high-performance corner
ax = axes[1]
for name, res in binary_results.items():
    proba_pos = res['proba'][:, ppe_present_idx]
    fpr, tpr, _ = roc_curve(yteb, proba_pos, pos_label=ppe_present_idx)
    roc_auc     = auc(fpr, tpr)
    col         = MODEL_COLORS.get(name, 'gray')
    ax.plot(fpr, tpr, lw=2.5, color=col,
            label=f"{name}  (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.2)
ax.set_xlim([0.0, 0.5]); ax.set_ylim([0.5, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("Zoomed — Top-Left Corner", fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
ax.annotate("Ideal classifier", xy=(0, 1), xytext=(0.05, 0.92),
            arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=10)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "report_roc_binary.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 5b.  MULTI-CLASS ROC CURVES (One-vs-Rest per model)
# ══════════════════════════════════════════════════════════════════════════════
# Binarise ground truth for OvR
yte_bin = label_binarize(yte, classes=list(range(N_CLASSES)))  # (N_test, 5)

model_order = ['SVM', 'RF', 'ExtraTrees', 'HistGBM', 'Ensemble', 'CNN']
fig, axes = plt.subplots(2, 3, figsize=(21, 14))
axes = axes.flatten()
fig.suptitle("Multi-class ROC Curves (One-vs-Rest) — All Models",
             fontsize=15, fontweight='bold', y=1.01)

class_linestyles = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(5,2))]

for ax_idx, model_name in enumerate(model_order):
    ax  = axes[ax_idx]
    res = multi_results[model_name]
    proba = res['proba']   # (N_test, 5)

    mean_tpr = np.zeros(200)
    mean_fpr = np.linspace(0, 1, 200)
    auc_scores = {}

    for cls_idx, cls_name in enumerate(ALL_CLASSES):
        col = CLASS_COLORS.get(cls_name, 'gray')
        fpr, tpr, _ = roc_curve(yte_bin[:, cls_idx], proba[:, cls_idx])
        roc_auc     = auc(fpr, tpr)
        auc_scores[cls_name] = roc_auc
        ls = class_linestyles[cls_idx % len(class_linestyles)]
        ax.plot(fpr, tpr, lw=2, color=col, linestyle=ls,
                label=f"{cls_name}  (AUC={roc_auc:.3f})")
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    mean_tpr /= N_CLASSES
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, 'k-', lw=3, alpha=0.85,
            label=f"Macro avg  (AUC={mean_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("FPR", fontsize=10); ax.set_ylabel("TPR", fontsize=10)
    col_title = MODEL_COLORS.get(model_name, '#333333')
    ax.set_title(f"{model_name}  (acc={res['accuracy']:.3f})",
                 fontsize=12, fontweight='bold', color=col_title)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "report_roc_multiclass.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 5c.  AUC SUMMARY HEATMAP — macro AUC per model per class
# ══════════════════════════════════════════════════════════════════════════════
auc_table = {}
for model_name in model_order:
    res   = multi_results[model_name]
    proba = res['proba']
    row   = {}
    for cls_idx, cls_name in enumerate(ALL_CLASSES):
        fpr, tpr, _ = roc_curve(yte_bin[:, cls_idx], proba[:, cls_idx])
        row[cls_name] = round(auc(fpr, tpr), 4)
    # macro avg
    row['Macro Avg'] = round(np.mean(list(row.values())), 4)
    auc_table[model_name] = row

auc_df = pd.DataFrame(auc_table).T   # models as rows, classes as cols

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(auc_df, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.70, vmax=1.00, ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'AUC', 'shrink': 0.8})
ax.set_title("AUC Scores — Multi-class OvR (per Model × Class)",
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Class", fontsize=11); ax.set_ylabel("Model", fontsize=11)
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "report_auc_heatmap.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 5d.  PRECISION-RECALL CURVES (multi-class OvR)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(21, 14))
axes = axes.flatten()
fig.suptitle("Precision-Recall Curves (One-vs-Rest) — All Models",
             fontsize=15, fontweight='bold', y=1.01)

for ax_idx, model_name in enumerate(model_order):
    ax    = axes[ax_idx]
    res   = multi_results[model_name]
    proba = res['proba']

    for cls_idx, cls_name in enumerate(ALL_CLASSES):
        col = CLASS_COLORS.get(cls_name, 'gray')
        precision, recall, _ = precision_recall_curve(
            yte_bin[:, cls_idx], proba[:, cls_idx])
        ap = average_precision_score(yte_bin[:, cls_idx], proba[:, cls_idx])
        ls = class_linestyles[cls_idx % len(class_linestyles)]
        ax.plot(recall, precision, lw=2, color=col, linestyle=ls,
                label=f"{cls_name}  (AP={ap:.3f})")

    # Baseline (random)
    class_freq = yte_bin.mean(axis=0)
    avg_freq   = class_freq.mean()
    ax.axhline(avg_freq, color='gray', lw=1.2, ls='--',
               label=f"Random ({avg_freq:.2f})")

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall", fontsize=10); ax.set_ylabel("Precision", fontsize=10)
    col_title = MODEL_COLORS.get(model_name, '#333333')
    ax.set_title(f"{model_name}  (acc={res['accuracy']:.3f})",
                 fontsize=12, fontweight='bold', color=col_title)
    ax.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "report_pr_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 5e.  ROC OVERLAY — best models on one chart for easy comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, N_CLASSES, figsize=(5 * N_CLASSES, 6), sharey=True)
fig.suptitle("Per-Class OvR ROC Comparison — All Models",
             fontsize=14, fontweight='bold')

for cls_idx, cls_name in enumerate(ALL_CLASSES):
    ax = axes[cls_idx]
    for model_name in model_order:
        res   = multi_results[model_name]
        proba = res['proba']
        fpr, tpr, _ = roc_curve(yte_bin[:, cls_idx], proba[:, cls_idx])
        roc_auc     = auc(fpr, tpr)
        col = MODEL_COLORS.get(model_name, 'gray')
        ax.plot(fpr, tpr, lw=2, color=col,
                label=f"{model_name}  ({roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("FPR", fontsize=10)
    if cls_idx == 0:
        ax.set_ylabel("TPR", fontsize=10)
    col_cls = CLASS_COLORS.get(cls_name, '#333333')
    ax.set_title(cls_name.upper(), fontsize=11, fontweight='bold', color=col_cls)
    ax.legend(loc='lower right', fontsize=7.5)
    ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "report_roc_per_class.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 5f.  CONFUSION MATRICES — all models
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
axes = axes.flatten()
fig.suptitle("Confusion Matrices — All Multi-class Models",
             fontsize=15, fontweight='bold')

for ax_idx, model_name in enumerate(model_order):
    ax  = axes[ax_idx]
    res = multi_results[model_name]
    disp = ConfusionMatrixDisplay(res['cm'], display_labels=ALL_CLASSES)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d',
              xticks_rotation=35)
    col_title = MODEL_COLORS.get(model_name, '#333333')
    ax.set_title(f"{model_name}  (acc={res['accuracy']:.3f})",
                 fontsize=12, fontweight='bold', color=col_title)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "report_confusion_all.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 5g.  PER-CLASS F1 HEATMAP — all models
# ══════════════════════════════════════════════════════════════════════════════
f1_data = {}
for model_name in model_order:
    rpt = multi_results[model_name]['report']
    f1_data[model_name] = {cls: round(rpt.get(cls, {}).get('f1-score', 0), 4)
                           for cls in ALL_CLASSES}
    f1_data[model_name]['Macro F1'] = round(rpt['macro avg']['f1-score'], 4)

f1_df = pd.DataFrame(f1_data).T

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(f1_df, annot=True, fmt='.3f', cmap='YlOrRd',
            vmin=0.40, vmax=1.00, ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'F1 Score', 'shrink': 0.8})
ax.set_title("Per-class F1 Scores — All Production Models",
             fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Class", fontsize=11); ax.set_ylabel("Model", fontsize=11)
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "report_f1_heatmap.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 5h.  MODEL COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# --- Multi-class accuracy ---
ax = axes[0]
names  = model_order
accs   = [multi_results[m]['accuracy'] for m in model_order]
colors = [MODEL_COLORS[m] for m in model_order]
macro_f1s = [multi_results[m]['report']['macro avg']['f1-score'] for m in model_order]

x = np.arange(len(names))
bars1 = ax.bar(x - 0.22, accs,     0.4, color=colors, alpha=0.9, edgecolor='black', label='Accuracy')
bars2 = ax.bar(x + 0.22, macro_f1s, 0.4, color=colors, alpha=0.55, edgecolor='black',
               hatch='//', label='Macro F1')

ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=10)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=11)
ax.set_title("Multi-class: Accuracy vs Macro F1", fontsize=13, fontweight='bold')
ax.axhline(0.8, color='red', lw=1.2, ls='--', alpha=0.6, label='0.80 threshold')
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=9)
for bar, val in zip(bars1, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, macro_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# --- Binary accuracy ---
ax = axes[1]
bin_names  = list(binary_results.keys())
bin_accs   = [binary_results[m]['accuracy'] for m in bin_names]
bin_colors = [MODEL_COLORS[m] for m in bin_names]
bin_f1s    = [binary_results[m]['report']['macro avg']['f1-score'] for m in bin_names]

xb = np.arange(len(bin_names))
b1 = ax.bar(xb - 0.22, bin_accs, 0.4, color=bin_colors, alpha=0.9, edgecolor='black', label='Accuracy')
b2 = ax.bar(xb + 0.22, bin_f1s,  0.4, color=bin_colors, alpha=0.55, edgecolor='black',
            hatch='//', label='Macro F1')

ax.set_xticks(xb); ax.set_xticklabels(bin_names, rotation=20, fontsize=10)
ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=11)
ax.set_title("Binary (PPE Present/Absent): Accuracy vs Macro F1", fontsize=13, fontweight='bold')
ax.axhline(0.8, color='red', lw=1.2, ls='--', alpha=0.6)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=9)
for bar, val in zip(b1, bin_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(b2, bin_f1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle("PPE Detection — Model Performance Comparison\n"
             f"Dataset: {len(crops_rgb):,} crops (MinhNKB + Jomarkow) | Test size: 20%",
             fontsize=14, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "report_model_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  COMPREHENSIVE SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Building summary dashboard ...")

fig = plt.figure(figsize=(24, 22))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.40,
                        left=0.05, right=0.97, top=0.93, bottom=0.04)

BG    = '#161b22'
BORD  = '#30363d'
TITLE_C = '#f0b429'
TEXT_C  = '#e6edf3'
MUTED   = '#8b949e'

def sax(ax, title=""):
    ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_edgecolor(BORD)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    if title:
        ax.set_title(title, color=TITLE_C, fontsize=10, fontweight='bold', pad=7)

# Header
ax_hdr = fig.add_subplot(gs[0, :])
ax_hdr.set_facecolor('#0d1117'); ax_hdr.axis('off')
ax_hdr.text(0.5, 0.78, "PPE DETECTION SYSTEM — FULL EVALUATION REPORT",
            transform=ax_hdr.transAxes, ha='center', va='center',
            fontsize=22, fontweight='bold', color=TITLE_C, fontfamily='monospace')
ax_hdr.text(0.5, 0.42,
            "6 Models Evaluated  •  SVM | RandomForest | ExtraTrees | HistGBM | Ensemble | CNN PPENet",
            transform=ax_hdr.transAxes, ha='center', va='center',
            fontsize=12, color=TEXT_C)
ax_hdr.text(0.5, 0.12,
            f"Dataset: {len(crops_rgb):,} crops (MinhNKB + Jomarkow)  •  Classes: {', '.join(ALL_CLASSES)}  •  Test split: 20%",
            transform=ax_hdr.transAxes, ha='center', va='center',
            fontsize=9, color=MUTED)

# Row 1 — Multi-class accuracy
ax1 = fig.add_subplot(gs[1, :2])
sax(ax1, "Multi-class Accuracy (5 classes)")
x = np.arange(len(model_order))
bars = ax1.bar(x, [multi_results[m]['accuracy'] for m in model_order],
               color=[MODEL_COLORS[m] for m in model_order],
               edgecolor=BORD, width=0.6)
ax1.set_xticks(x); ax1.set_xticklabels(model_order, rotation=18, fontsize=8)
ax1.set_ylim(0, 1.12); ax1.set_ylabel("Accuracy", color=TEXT_C)
ax1.axhline(0.8, color=MUTED, lw=1, ls='--', alpha=0.6)
ax1.grid(axis='y', alpha=0.2)
for bar, val in zip(bars, [multi_results[m]['accuracy'] for m in model_order]):
    ax1.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.3f}',
             ha='center', va='bottom', color=TEXT_C, fontsize=9, fontweight='bold')

# Row 1 — Binary accuracy
ax2 = fig.add_subplot(gs[1, 2:])
sax(ax2, "Binary Accuracy (PPE Present vs No PPE)")
x2 = np.arange(len(bin_names))
bars2 = ax2.bar(x2, bin_accs, color=[MODEL_COLORS[m] for m in bin_names],
                edgecolor=BORD, width=0.6)
ax2.set_xticks(x2); ax2.set_xticklabels(bin_names, rotation=18, fontsize=8)
ax2.set_ylim(0, 1.12); ax2.set_ylabel("Accuracy", color=TEXT_C)
ax2.axhline(0.8, color=MUTED, lw=1, ls='--', alpha=0.6)
ax2.grid(axis='y', alpha=0.2)
for bar, val in zip(bars2, bin_accs):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.3f}',
             ha='center', va='bottom', color=TEXT_C, fontsize=9, fontweight='bold')

# Row 2 — Binary ROC mini (dashboard)
ax3 = fig.add_subplot(gs[2, :2])
sax(ax3, "ROC Curves — Binary Models")
for name, res in binary_results.items():
    proba_pos = res['proba'][:, ppe_present_idx]
    fpr, tpr, _ = roc_curve(yteb, proba_pos, pos_label=ppe_present_idx)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, lw=2, color=MODEL_COLORS.get(name,'gray'),
             label=f"{name} AUC={roc_auc:.3f}")
ax3.plot([0,1],[0,1],'w--',lw=1,alpha=0.4)
ax3.set_xlabel("FPR",color=TEXT_C); ax3.set_ylabel("TPR",color=TEXT_C)
ax3.legend(loc='lower right', fontsize=8, facecolor=BG, edgecolor=BORD, labelcolor=TEXT_C)
ax3.grid(alpha=0.2)

# Row 2 — AUC heatmap mini
ax4 = fig.add_subplot(gs[2, 2:])
sax(ax4, "AUC Heatmap (Multi-class OvR)")
sns.heatmap(auc_df, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.70, vmax=1.00, ax=ax4,
            linewidths=0.4, linecolor=BORD,
            cbar_kws={'label':'AUC','shrink':0.6},
            annot_kws={'size':7})
ax4.tick_params(axis='x', rotation=25, labelsize=7)
ax4.tick_params(axis='y', labelsize=7)
ax4.set_title("AUC Heatmap (Multi-class OvR)", color=TITLE_C,
              fontsize=10, fontweight='bold', pad=7)

# Row 3 — Findings table
ax5 = fig.add_subplot(gs[3, :])
ax5.set_facecolor(BG); ax5.axis('off')
ax5.set_title("Evaluation Summary — Key Findings", color=TITLE_C,
              fontsize=10, fontweight='bold', pad=7)

best_multi  = max(model_order, key=lambda m: multi_results[m]['accuracy'])
best_binary = max(bin_names,   key=lambda m: binary_results[m]['accuracy'])
best_multi_acc   = multi_results[best_multi]['accuracy']
best_binary_acc  = binary_results[best_binary]['accuracy']

# Compute macro AUC per model for summary
def macro_auc(model_name):
    proba = multi_results[model_name]['proba']
    aucs  = []
    for ci in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(yte_bin[:, ci], proba[:, ci])
        aucs.append(auc(fpr, tpr))
    return np.mean(aucs)

best_auc_model = max(model_order, key=macro_auc)
best_auc_val   = macro_auc(best_auc_model)

findings = [
    ("BEST MULTI-CLASS",   f"{best_multi}  ({best_multi_acc:.1%} accuracy)",               '#27ae60'),
    ("BEST BINARY",        f"{best_binary}  ({best_binary_acc:.1%} accuracy)",              '#3498db'),
    ("BEST MACRO AUC",     f"{best_auc_model}  (AUC={best_auc_val:.3f})",                   '#9b59b6'),
    ("CNN PERFORMANCE",    f"87.3% val acc, 100 epochs, 64×64 crops, AdamW+OneCycleLR",    '#1abc9c'),
    ("HARDEST CLASS",      "partial_ppe (lowest F1 across all models)",                     '#e74c3c'),
    ("EASIEST CLASS",      "helmet & safety_vest (highest F1 ≈ 0.93-0.95 CNN)",            '#27ae60'),
    ("DATASET",            "3,626 crops: MinhNKB (5 cls) + Jomarkow (2 cls), 20% test",   '#f39c12'),
    ("FEATURES (ML)",      "HOG 1764-dim + 6×32 color hist = 1956-dim per crop",           '#e67e22'),
    ("BINARY TASK",        "PPE present (helmet/vest/full/partial) vs no_ppe",              '#3498db'),
    ("DEPLOYMENT NOTE",    "Two-stage: YOLOv8n person detect → PPENetFast classify",       '#f0b429'),
]

for i, (label, text, color) in enumerate(findings):
    y = 0.97 - i * 0.098
    ax5.text(0.01, y, f"  {label}:", transform=ax5.transAxes,
             color=color, fontsize=8.5, fontweight='bold', va='top')
    ax5.text(0.18, y, text, transform=ax5.transAxes,
             color=TEXT_C, fontsize=8.5, va='top')

out_path = os.path.join(OUT_DIR, "report_dashboard.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"  Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  PRINT FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Final summary ...")
print("\n" + "="*75)
print("PPE DETECTION — FULL EVALUATION REPORT")
print("="*75)

print("\n-- MULTI-CLASS RESULTS (5 classes) --")
hdr = f"{'Model':<20} {'Accuracy':>9} {'Macro F1':>10} {'Weighted F1':>12} {'Macro AUC':>10}"
print(hdr)
print("-"*63)
for m in model_order:
    res = multi_results[m]
    rpt = res['report']
    mauc = macro_auc(m)
    print(f"{m:<20} {res['accuracy']:>9.4f} "
          f"{rpt['macro avg']['f1-score']:>10.4f} "
          f"{rpt['weighted avg']['f1-score']:>12.4f} "
          f"{mauc:>10.4f}")

print("\n-- BINARY RESULTS (ppe_present vs no_ppe) --")
hdr2 = f"{'Model':<20} {'Accuracy':>9} {'Macro F1':>10} {'AUC':>8}"
print(hdr2)
print("-"*50)
for m in bin_names:
    res = binary_results[m]
    rpt = res['report']
    proba_pos = res['proba'][:, ppe_present_idx]
    fpr, tpr, _ = roc_curve(yteb, proba_pos, pos_label=ppe_present_idx)
    b_auc = auc(fpr, tpr)
    print(f"{m:<20} {res['accuracy']:>9.4f} "
          f"{rpt['macro avg']['f1-score']:>10.4f} "
          f"{b_auc:>8.4f}")

print("\n-- PER-CLASS F1 (CNN) --")
for cls in ALL_CLASSES:
    f1 = cnn_rpt.get(cls, {}).get('f1-score', 0)
    print(f"  {cls:<15}: F1 = {f1:.4f}")

print("\n-- AUC SUMMARY TABLE --")
print(auc_df.to_string())

# Save CSV summary
summary_rows = []
for m in model_order:
    res  = multi_results[m]
    rpt  = res['report']
    mauc = macro_auc(m)
    per_class_f1 = {cls: round(rpt.get(cls,{}).get('f1-score',0),4) for cls in ALL_CLASSES}
    row = {'Model': m, 'Task': 'multi',
           'Accuracy': round(res['accuracy'], 4),
           'Macro_F1': round(rpt['macro avg']['f1-score'], 4),
           'Weighted_F1': round(rpt['weighted avg']['f1-score'], 4),
           'Macro_AUC': round(mauc, 4)}
    row.update({f'F1_{c}': v for c, v in per_class_f1.items()})
    summary_rows.append(row)

for m in bin_names:
    res  = binary_results[m]
    rpt  = res['report']
    proba_pos = res['proba'][:, ppe_present_idx]
    fpr, tpr, _ = roc_curve(yteb, proba_pos, pos_label=ppe_present_idx)
    b_auc = auc(fpr, tpr)
    row = {'Model': m, 'Task': 'binary',
           'Accuracy': round(res['accuracy'], 4),
           'Macro_F1': round(rpt['macro avg']['f1-score'], 4),
           'Weighted_F1': round(rpt['weighted avg']['f1-score'], 4),
           'AUC': round(b_auc, 4)}
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
csv_path = os.path.join(OUT_DIR, "report_full_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\nSaved CSV: {csv_path}")

# List all report outputs
print("\n" + "="*75)
print(f"ALL REPORT FILES IN: {OUT_DIR}")
print("="*75)
report_files = [f for f in sorted(os.listdir(OUT_DIR)) if f.startswith('report_')]
for f in report_files:
    sz = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f:<45} {sz/1024:7.1f} KB")
print("="*75)
print("Report generation complete.")
