"""
PPE Pipeline Evaluation Report Generator
=========================================
Generates a self-contained HTML report with:
  - Multi-class one-vs-rest ROC/AUC for every ML model + CNN
  - Binary ROC/AUC for binary classifiers
  - Precision-Recall curves
  - Model comparison, confusion matrices, F1 heatmap
  - CCTV validation summary

Usage (from project root):
    python skills/ppe-report-generator/scripts/generate_report.py
"""

import os, sys, time, warnings, base64, io
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
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR   = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(os.path.dirname(SKILL_DIR))   # PPE-Detection/
BASE        = os.path.dirname(PROJECT_DIR)                   # D:/Claude/

CACHE_DIR   = os.path.join(BASE, "cache")
MODELS_DIR  = os.path.join(PROJECT_DIR, "results", "models")
REPORT_DIR  = os.path.join(PROJECT_DIR, "results", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

ALL_CLASSES  = ["full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]
BINARY_CLASSES = ["no_ppe", "ppe_present"]
CLASS_COLORS = {
    'full_ppe':    '#8e44ad',
    'helmet':      '#27ae60',
    'no_ppe':      '#e74c3c',
    'partial_ppe': '#e67e22',
    'safety_vest': '#2980b9',
}
MODEL_PALETTE = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#95a5a6']

print("="*65)
print("PPE EVALUATION REPORT GENERATOR")
print("="*65)

# ── Load cache ─────────────────────────────────────────────────────
feat_path  = os.path.join(CACHE_DIR, "features_600.npy")
label_path = os.path.join(CACHE_DIR, "crops_y_600.npy")
crops_path = os.path.join(CACHE_DIR, "crops_X_600.npy")

if not os.path.exists(feat_path) or not os.path.exists(label_path):
    print(f"\nERROR: Cache not found at {CACHE_DIR}")
    print("Run 'python src/ppe_production_train.py' first to generate the cache.")
    sys.exit(1)

print(f"\n[1/5] Loading cached features...")
X_ml = np.load(feat_path)
y_raw = np.load(label_path)
print(f"  Features: {X_ml.shape}  |  Labels: {y_raw.shape}")
print(f"  Class distribution: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

# Encode labels — multi and binary
le_multi = LabelEncoder()
y_multi  = le_multi.fit_transform(y_raw)

binary_map = {c: ("ppe_present" if c != "no_ppe" else "no_ppe") for c in ALL_CLASSES}
y_bin_raw  = np.array([binary_map[l] for l in y_raw])
le_binary  = LabelEncoder()
y_binary   = le_binary.fit_transform(y_bin_raw)

# Same 80/20 split used during training
Xtr, Xte, ytr, yte = train_test_split(X_ml, y_multi, test_size=0.2,
                                        random_state=42, stratify=y_multi)
_, Xte_b, _, yte_b  = train_test_split(X_ml, y_binary, test_size=0.2,
                                        random_state=42, stratify=y_binary)

# Binarised labels for multi-class ROC
yte_bin = label_binarize(yte, classes=list(range(len(le_multi.classes_))))
print(f"  Test set: {len(Xte)} samples")

# ── Load ML models ─────────────────────────────────────────────────
print(f"\n[2/5] Loading models...")

MULTI_MODELS = [
    ("SVM",       "prod_svm_multi.pkl"),
    ("RF",        "prod_rf_multi.pkl"),
    ("ExtraTrees","prod_et_multi.pkl"),
    ("GBM",       "prod_gbm_multi.pkl"),
    ("Ensemble",  "prod_ensemble_multi.pkl"),
]
BINARY_MODELS = [
    ("SVM (bin)",  "prod_svm_binary.pkl"),
    ("RF (bin)",   "prod_rf_binary.pkl"),
    ("GBM (bin)",  "prod_gbm_binary.pkl"),
]

ml_multi, ml_binary = {}, {}
for name, fname in MULTI_MODELS:
    fp = os.path.join(MODELS_DIR, fname)
    if os.path.exists(fp):
        print(f"  Loading {name}...", end=' ', flush=True)
        t0 = time.time()
        ml_multi[name] = joblib.load(fp)
        print(f"{time.time()-t0:.1f}s")
    else:
        print(f"  Skipping {name} (not found)")

for name, fname in BINARY_MODELS:
    fp = os.path.join(MODELS_DIR, fname)
    if os.path.exists(fp):
        ml_binary[name] = joblib.load(fp)

# ── Load CNN ───────────────────────────────────────────────────────
cnn_model, le_cnn, X_cnn_te, y_cnn_te = None, None, None, None
cnn_path = os.path.join(MODELS_DIR, "prod_cnn_model.pth")
le_cnn_path = os.path.join(MODELS_DIR, "prod_le_cnn.pkl")
crops_exist = os.path.exists(crops_path)

if os.path.exists(cnn_path) and os.path.exists(le_cnn_path) and crops_exist:
    print(f"  Loading CNN...", end=' ', flush=True)
    t0 = time.time()

    class PPENet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
                nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
                nn.MaxPool2d(2),nn.Dropout2d(0.1),
                # Block 2
                nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                nn.MaxPool2d(2),nn.Dropout2d(0.1),
                # Block 3
                nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                nn.MaxPool2d(2),nn.Dropout2d(0.2),
                nn.AdaptiveAvgPool2d((1,1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128,256),nn.BatchNorm1d(256),nn.ReLU(inplace=True),nn.Dropout(0.4),
                nn.Linear(256,128),nn.ReLU(inplace=True),nn.Dropout(0.3),
                nn.Linear(128,num_classes),
            )
        def forward(self, x): return self.classifier(self.features(x))

    le_cnn = joblib.load(le_cnn_path)
    ckpt   = torch.load(cnn_path, map_location='cpu', weights_only=False)  # own checkpoint, safe
    cnn_model = PPENet(len(le_cnn.classes_))
    cnn_model.load_state_dict(ckpt['state_dict'])
    cnn_model.eval()

    crops_rgb = np.load(crops_path)
    y_cnn_all = le_cnn.transform(y_raw)
    _, cnn_te_idx = train_test_split(range(len(y_cnn_all)), test_size=0.2,
                                      random_state=42, stratify=y_cnn_all)
    X_arr = crops_rgb[cnn_te_idx].astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406],dtype=np.float32)
    std  = np.array([0.229,0.224,0.225],dtype=np.float32)
    X_arr = (X_arr - mean) / std
    X_cnn_te = torch.tensor(X_arr.transpose(0,3,1,2))
    y_cnn_te  = np.array(y_cnn_all)[cnn_te_idx]
    print(f"{time.time()-t0:.1f}s  ({len(cnn_te_idx)} test crops)")
else:
    print("  CNN: skipping (model or crops not found)")

# ── Compute ROC/AUC ────────────────────────────────────────────────
print(f"\n[3/5] Computing ROC/AUC curves...")

def get_proba_multi(model, X):
    return model.predict_proba(X)

def get_proba_cnn(X_t):
    all_probs = []
    batch = 256
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            out = cnn_model(X_t[i:i+batch])
            all_probs.append(torch.softmax(out, 1).numpy())
    return np.vstack(all_probs)

# Multi-class ROC storage: {model_name: {class_name: (fpr, tpr, auc_val)}}
roc_multi = {}
for name, model in ml_multi.items():
    print(f"  ROC: {name}...", end=' ', flush=True)
    proba = get_proba_multi(model, Xte)
    class_rocs = {}
    for i, cls in enumerate(le_multi.classes_):
        fpr, tpr, _ = roc_curve(yte_bin[:, i], proba[:, i])
        class_rocs[cls] = (fpr, tpr, auc(fpr, tpr))
    roc_multi[name] = class_rocs
    macro_auc = np.mean([v[2] for v in class_rocs.values()])
    print(f"macro AUC={macro_auc:.3f}")

# CNN ROC
if cnn_model is not None:
    print(f"  ROC: CNN...", end=' ', flush=True)
    proba_cnn = get_proba_cnn(X_cnn_te)
    y_cnn_bin = label_binarize(y_cnn_te, classes=list(range(len(le_cnn.classes_))))
    class_rocs = {}
    for i, cls in enumerate(le_cnn.classes_):
        fpr, tpr, _ = roc_curve(y_cnn_bin[:, i], proba_cnn[:, i])
        class_rocs[cls] = (fpr, tpr, auc(fpr, tpr))
    roc_multi["CNN"] = class_rocs
    macro_auc = np.mean([v[2] for v in class_rocs.values()])
    print(f"macro AUC={macro_auc:.3f}")

# Binary ROC
roc_binary = {}
for name, model in ml_binary.items():
    proba_b = model.predict_proba(Xte_b)[:, 1]
    fpr, tpr, _ = roc_curve(yte_b, proba_b)
    roc_binary[name] = (fpr, tpr, auc(fpr, tpr))

# ── Precision-Recall curves ────────────────────────────────────────
pr_multi = {}
for name, model in ml_multi.items():
    proba = get_proba_multi(model, Xte)
    class_prs = {}
    for i, cls in enumerate(le_multi.classes_):
        p, r, _ = precision_recall_curve(yte_bin[:, i], proba[:, i])
        ap = average_precision_score(yte_bin[:, i], proba[:, i])
        class_prs[cls] = (p, r, ap)
    pr_multi[name] = class_prs

if cnn_model is not None:
    proba_cnn = get_proba_cnn(X_cnn_te)
    class_prs = {}
    for i, cls in enumerate(le_cnn.classes_):
        p, r, _ = precision_recall_curve(y_cnn_bin[:, i], proba_cnn[:, i])
        ap = average_precision_score(y_cnn_bin[:, i], proba_cnn[:, i])
        class_prs[cls] = (p, r, ap)
    pr_multi["CNN"] = class_prs

# ── Generate plots ─────────────────────────────────────────────────
print(f"\n[4/5] Generating plots...")

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

plots = {}  # name -> base64 PNG

# 1. Multi-class ROC grid (one subplot per model)
n_models = len(roc_multi)
ncols = min(3, n_models)
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
if n_models == 1:
    axes = np.array([[axes]])
elif nrows == 1:
    axes = axes.reshape(1, -1)
fig.suptitle("Multi-Class ROC Curves (One-vs-Rest) — All Models",
             fontsize=14, fontweight='bold', y=1.01)

for ax, (model_name, class_rocs) in zip(axes.flatten(), roc_multi.items()):
    for cls, (fpr, tpr, auc_val) in class_rocs.items():
        ax.plot(fpr, tpr, lw=2, color=CLASS_COLORS.get(cls,'gray'),
                label=f"{cls} (AUC={auc_val:.3f})")
    macro_auc = np.mean([v[2] for v in class_rocs.values()])
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
    ax.set_title(f"{model_name}\nMacro AUC = {macro_auc:.3f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

for ax in axes.flatten()[n_models:]:
    ax.axis('off')

plt.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "roc_multiclass.png"), dpi=130, bbox_inches='tight')
plots['roc_multiclass'] = fig_to_b64(fig)
print("  roc_multiclass.png")

# 2. Binary ROC
fig, ax = plt.subplots(figsize=(7, 5))
for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_binary.items()):
    ax.plot(fpr, tpr, lw=2.5, color=MODEL_PALETTE[i],
            label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_title("Binary ROC Curves — PPE Present vs No PPE", fontsize=12, fontweight='bold')
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
plt.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "roc_binary.png"), dpi=130, bbox_inches='tight')
plots['roc_binary'] = fig_to_b64(fig)
print("  roc_binary.png")

# 3. Macro AUC comparison bar chart
macro_aucs = {name: np.mean([v[2] for v in cr.values()])
              for name, cr in roc_multi.items()}
fig, ax = plt.subplots(figsize=(9, 4))
names, aucs = list(macro_aucs.keys()), list(macro_aucs.values())
bars = ax.bar(names, aucs, color=MODEL_PALETTE[:len(names)], edgecolor='black', width=0.6)
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
ax.axhline(0.9, color='green', lw=1.5, linestyle='--', alpha=0.6, label='AUC=0.90')
ax.axhline(0.8, color='red',   lw=1.5, linestyle='--', alpha=0.4, label='AUC=0.80')
ax.set_ylim(0, 1.05); ax.set_ylabel("Macro AUC (one-vs-rest)")
ax.set_title("Macro-Averaged AUC — Multi-Class Models", fontsize=12, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plots['macro_auc_bar'] = fig_to_b64(fig)
print("  macro_auc_bar")

# 4. Per-class AUC heatmap
cls_names = list(le_multi.classes_)
auc_data = {}
for name, class_rocs in roc_multi.items():
    auc_data[name] = [class_rocs.get(c, (None,None,0))[2] for c in cls_names]
auc_df = pd.DataFrame(auc_data, index=cls_names)
fig, ax = plt.subplots(figsize=(max(8, len(roc_multi)*2), 4))
sns.heatmap(auc_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, vmin=0.5, vmax=1.0, cbar_kws={'label':'AUC'})
ax.set_title("Per-Class AUC (One-vs-Rest) — All Models", fontsize=12, fontweight='bold')
plt.tight_layout()
plots['per_class_auc_heatmap'] = fig_to_b64(fig)
print("  per_class_auc_heatmap")

# 5. Precision-Recall curves — one subplot per model
n_pr = len(pr_multi)
ncols_pr = min(3, n_pr)
nrows_pr = (n_pr + ncols_pr - 1) // ncols_pr
fig, axes = plt.subplots(nrows_pr, ncols_pr, figsize=(6*ncols_pr, 5*nrows_pr))
if n_pr == 1:
    axes = np.array([[axes]])
elif nrows_pr == 1:
    axes = axes.reshape(1, -1)
fig.suptitle("Precision-Recall Curves (One-vs-Rest) — All Models",
             fontsize=14, fontweight='bold', y=1.01)
for ax, (model_name, class_prs) in zip(axes.flatten(), pr_multi.items()):
    for cls, (p, r, ap) in class_prs.items():
        ax.plot(r, p, lw=2, color=CLASS_COLORS.get(cls,'gray'),
                label=f"{cls} (AP={ap:.3f})")
    macro_ap = np.mean([v[2] for v in class_prs.values()])
    ax.set_title(f"{model_name}\nMacro AP = {macro_ap:.3f}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=7, loc='lower left'); ax.grid(alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
for ax in axes.flatten()[n_pr:]:
    ax.axis('off')
plt.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "pr_curves.png"), dpi=130, bbox_inches='tight')
plots['pr_curves'] = fig_to_b64(fig)
print("  pr_curves.png")

# 6. Embed existing plots from results/models/
existing_plots = {
    'cnn_training':      'prod_cnn_training.png',
    'model_comparison':  'prod_model_comparison.png',
    'confusion_matrices':'prod_confusion_matrices.png',
    'cnn_confusion':     'prod_cnn_confusion.png',
    'f1_heatmap':        'prod_f1_heatmap.png',
    'cctv_validation':   'prod_cctv_validation.png',
}
for key, fname in existing_plots.items():
    fp = os.path.join(MODELS_DIR, fname)
    if os.path.exists(fp):
        with open(fp, 'rb') as f:
            plots[key] = base64.b64encode(f.read()).decode()

# ── Load model summary + CCTV results ─────────────────────────────
summary_path = os.path.join(MODELS_DIR, "prod_model_summary.csv")
summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()

cctv_path = os.path.join(MODELS_DIR, "prod_cctv_results.csv")
cctv_df = pd.DataFrame()
if os.path.exists(cctv_path):
    cctv_df = pd.read_csv(cctv_path).drop_duplicates()

# ── Build HTML report ──────────────────────────────────────────────
print(f"\n[5/5] Building HTML report...")

def df_to_html(df, highlight_col=None):
    if df.empty:
        return "<p><em>No data available.</em></p>"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                cell = f"<td>{val:.4f}</td>"
            else:
                cell = f"<td>{val}</td>"
            cells.append(cell)
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

def img_tag(b64, alt="plot", max_width="100%"):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:{max_width};border-radius:8px;margin:8px 0;">'

# AUC summary table
auc_rows = []
for model_name, class_rocs in roc_multi.items():
    row = {"Model": model_name}
    for cls in cls_names:
        row[cls] = f"{class_rocs.get(cls,(0,0,0))[2]:.3f}"
    row["Macro AUC"] = f"{np.mean([class_rocs.get(c,(0,0,0))[2] for c in cls_names]):.3f}"
    auc_rows.append(row)
auc_df_html = pd.DataFrame(auc_rows)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PPE Detection Pipeline — Evaluation Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 0; }}
  h1 {{ background: linear-gradient(90deg,#1f6feb,#8b5cf6); padding: 28px 40px; margin: 0; font-size: 2em; }}
  h2 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-top: 40px; }}
  h3 {{ color: #79c0ff; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 40px; }}
  .meta {{ color: #8b949e; font-size: 0.9em; margin-top: 6px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 20px 0; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th {{ background: #21262d; color: #58a6ff; padding: 10px; text-align: left; border-bottom: 2px solid #30363d; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.8em; font-weight:bold; }}
  .badge-cnn {{ background:#8b5cf6; }}
  .badge-ml  {{ background:#1f6feb; }}
  .section-note {{ color: #8b949e; font-size: 0.88em; margin-bottom: 12px; }}
  img {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>PPE Detection Pipeline — Evaluation Report</h1>
<div class="container">
<p class="meta">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp;
Dataset: 3,626 crops &nbsp;|&nbsp; Test split: 20% (stratified, seed=42)</p>

<!-- EXECUTIVE SUMMARY -->
<div class="card">
<h2>Executive Summary</h2>
<div class="grid-2">
  <div>
    <h3>Best Multi-Class Model</h3>
    <p><strong style="font-size:1.4em;color:#2ecc71;">CNN PPENet</strong>
    <span class="badge badge-cnn">CNN</span><br>
    Accuracy: <strong>{summary_df[summary_df['Task']=='multi']['Accuracy'].max():.1%}</strong> &nbsp;|&nbsp;
    Macro AUC: <strong>{max(np.mean([v[2] for v in cr.values()]) for cr in roc_multi.values()):.3f}</strong>
    </p>
  </div>
  <div>
    <h3>Best Binary Classifier</h3>
    <p><strong style="font-size:1.4em;color:#3498db;">SVM (PCA→RBF)</strong>
    <span class="badge badge-ml">ML</span><br>
    Accuracy: <strong>{summary_df[summary_df['Task']=='binary']['Accuracy'].max():.1%}</strong> &nbsp;|&nbsp;
    AUC: <strong>{max(v[2] for v in roc_binary.values()):.3f}</strong>
    </p>
  </div>
</div>
</div>

<!-- MODEL PERFORMANCE SUMMARY -->
<div class="card">
<h2>Model Performance Summary</h2>
{df_to_html(summary_df)}
</div>

<!-- AUC SUMMARY TABLE -->
<div class="card">
<h2>Per-Class AUC (One-vs-Rest)</h2>
<p class="section-note">AUC computed on the same 20% test split used during training (random_state=42, stratified).</p>
{df_to_html(auc_df_html)}
</div>

<!-- MACRO AUC BAR -->
<div class="card">
<h2>Macro AUC Comparison</h2>
{img_tag(plots['macro_auc_bar'], 'Macro AUC bar chart')}
</div>

<!-- MULTI-CLASS ROC -->
<div class="card">
<h2>Multi-Class ROC Curves (One-vs-Rest)</h2>
<p class="section-note">Each subplot shows the five class-specific ROC curves for one model.
A diagonal line represents a random classifier (AUC=0.50).</p>
{img_tag(plots['roc_multiclass'], 'Multi-class ROC curves')}
</div>

<!-- PER-CLASS AUC HEATMAP -->
<div class="card">
<h2>Per-Class AUC Heatmap</h2>
{img_tag(plots['per_class_auc_heatmap'], 'Per-class AUC heatmap')}
</div>

<!-- PRECISION-RECALL -->
<div class="card">
<h2>Precision-Recall Curves</h2>
<p class="section-note">PR curves complement ROC curves — particularly informative for imbalanced classes like full_ppe (426 samples vs 1000 for others).</p>
{img_tag(plots['pr_curves'], 'Precision-recall curves')}
</div>

<!-- BINARY ROC -->
<div class="card">
<h2>Binary ROC Curves (PPE Present vs No PPE)</h2>
{img_tag(plots['roc_binary'], 'Binary ROC curves')}
</div>

<!-- EXISTING PLOTS -->
{'<div class="card"><h2>CNN Training History</h2>' + img_tag(plots["cnn_training"]) + '</div>' if 'cnn_training' in plots else ''}
{'<div class="card"><h2>Model Accuracy Comparison</h2>' + img_tag(plots["model_comparison"]) + '</div>' if 'model_comparison' in plots else ''}
<div class="grid-2">
{'<div class="card"><h2>Confusion Matrices (ML Models)</h2>' + img_tag(plots["confusion_matrices"]) + '</div>' if 'confusion_matrices' in plots else ''}
{'<div class="card"><h2>CNN Confusion Matrix</h2>' + img_tag(plots["cnn_confusion"]) + '</div>' if 'cnn_confusion' in plots else ''}
</div>
{'<div class="card"><h2>Per-Class F1 Heatmap</h2>' + img_tag(plots["f1_heatmap"]) + '</div>' if 'f1_heatmap' in plots else ''}

<!-- CCTV VALIDATION -->
<div class="card">
<h2>CCTV Validation Results</h2>
{'<p class="section-note">Two-stage detection: HOG person detector + CNN classifier on real construction-site CCTV frames.</p>' + df_to_html(cctv_df) if not cctv_df.empty else '<p class="section-note">No CCTV results found.</p>'}
{'<br>' + img_tag(plots["cctv_validation"]) if 'cctv_validation' in plots else ''}
</div>

</div>
</body>
</html>
"""

out_path = os.path.join(REPORT_DIR, "ppe_evaluation_report.html")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"\n{'='*65}")
print(f"REPORT COMPLETE")
print(f"{'='*65}")
print(f"  HTML report : {out_path}")
print(f"  Size        : {os.path.getsize(out_path)/1024/1024:.1f} MB")
print(f"\n  Macro AUC summary:")
for model_name, class_rocs in roc_multi.items():
    macro = np.mean([v[2] for v in class_rocs.values()])
    print(f"    {model_name:20s}  {macro:.4f}")
print(f"\n  Binary AUC summary:")
for name, (_, _, auc_val) in roc_binary.items():
    print(f"    {name:20s}  {auc_val:.4f}")
