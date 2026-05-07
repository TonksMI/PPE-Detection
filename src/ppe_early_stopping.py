"""
PPE CNN WITH EARLY STOPPING
============================
Re-trains PPENet (64x64 CNN) with early stopping (patience=10) to establish
a cleaner reproducible baseline showing where validation accuracy naturally
stabilises.

Architecture: PPENet (NOT PPENetFast/SmallCNN)
  - 3 conv blocks: 32→64→128 filters with BatchNorm + Dropout
  - GlobalAvgPool → FC 128→256→128→num_classes
  - Input: 64×64 RGB crops

Outputs (all to results/models/):
  es_cnn_model.pth      — best weights + metadata
  es_cnn_training.png   — loss/accuracy curves with stopped_epoch dashed line
  es_cnn_confusion.png  — confusion matrix
  es_cnn_results.csv    — single-row metrics summary

Usage:
  python src/ppe_early_stopping.py
"""

import os
import sys
import glob
import random
import time
import warnings
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')   # MUST be before any other matplotlib/pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# ── Reproducibility ────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── Device ─────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Path resolution (Windows / Linux auto-detect) ───────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)   # PPE-Detection/
BASE        = os.path.dirname(PROJECT_DIR)  # D:\Claude or /sessions/...

if os.path.exists("D:/datasets/jomarkow"):
    DATASETS = "D:/datasets"
else:
    DATASETS = os.path.join(BASE, "datasets")

MINHNKB_IMG = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-data")
MINHNKB_ANN = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-annotations-new")
JOMARK_IMG  = os.path.join(DATASETS, "jomarkow/images")
JOMARK_LBL  = os.path.join(DATASETS, "jomarkow/labels")

CACHE_DIR = os.path.join(BASE, "cache")
OUT_DIR   = os.path.join(PROJECT_DIR, "results/models")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# ── Dataset / training hyperparameters ─────────────────────────
MAX_CLASS = 600          # crops per class ceiling (matches production cache)
BATCH     = 128          # batch size
MAX_EPOCHS = 100         # upper bound — early stopping may terminate sooner
PATIENCE  = 10           # early stopping patience

ALL_CLASSES  = ["full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]
MINHNKB_MAP  = {
    "helmet":                     "helmet",
    "safety vest":                "safety_vest",
    "person with full safety":    "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety":      "no_ppe",
}
JOMARKOW_MAP = {0: "helmet", 1: "no_ppe"}

num_workers = 0   # Windows DataLoader constraint

print("=" * 65)
print("PPE CNN — EARLY STOPPING TRAINING")
print(f"Device : {DEVICE}")
print(f"Max epochs : {MAX_EPOCHS}  |  Patience : {PATIENCE}  |  Batch : {BATCH}")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (cache → fast path; else dual-dataset parse)
# ══════════════════════════════════════════════════════════════════
CACHE_X = os.path.join(CACHE_DIR, f"crops_X_{MAX_CLASS}.npy")
CACHE_Y = os.path.join(CACHE_DIR, f"crops_y_{MAX_CLASS}.npy")

if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
    print(f"\n[1/4] Loading cached crops from {CACHE_DIR} ...")
    crops_rgb  = np.load(CACHE_X)
    labels_raw = np.load(CACHE_Y)
    print(f"  Loaded {len(crops_rgb)} crops")
else:
    print(f"\n[1/4] Parsing datasets (no cache found) ...")
    t0 = time.time()
    crops_rgb, labels_raw = [], []
    cc = {c: 0 for c in ALL_CLASSES}

    # ── MinhNKB dataset (Pascal VOC XML) ────────────────────────
    xml_files = sorted(glob.glob(os.path.join(MINHNKB_ANN, "*.xml")))
    random.shuffle(xml_files)
    for xf in xml_files:
        try:
            root  = ET.parse(xf).getroot()
            fname = root.findtext("filename")
            ip    = os.path.join(MINHNKB_IMG, fname) if fname else None
            if not ip or not os.path.exists(ip):
                stem = os.path.splitext(os.path.basename(xf))[0]
                for e in [".jpg", ".jpeg", ".png"]:
                    c = os.path.join(MINHNKB_IMG, stem + e)
                    if os.path.exists(c):
                        ip = c
                        break
            if not ip or not os.path.exists(ip):
                continue
            img = cv2.imread(ip)
            if img is None:
                continue
            sz = root.find("size")
            iw = int(sz.findtext("width",  0))
            ih = int(sz.findtext("height", 0))
            for obj in root.findall("object"):
                raw = obj.findtext("name", "").strip().lower()
                if raw not in MINHNKB_MAP:
                    continue
                cls = MINHNKB_MAP[raw]
                if cc[cls] >= MAX_CLASS:
                    continue
                bb = obj.find("bndbox")
                x1 = max(0,  int(float(bb.findtext("xmin", 0))))
                y1 = max(0,  int(float(bb.findtext("ymin", 0))))
                x2 = min(iw, int(float(bb.findtext("xmax", iw))))
                y2 = min(ih, int(float(bb.findtext("ymax", ih))))
                if x2 > x1 + 8 and y2 > y1 + 8:
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crops_rgb.append(
                        cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2RGB)
                    )
                    labels_raw.append(cls)
                    cc[cls] += 1
        except Exception:
            pass
    print(f"  MinhNKB: {len(crops_rgb)} crops | {cc}")

    # ── Jomarkow dataset (YOLO TXT) ─────────────────────────────
    cc_j  = {0: 0, 1: 0}
    MAX_J = min(MAX_CLASS, 400)
    lbl_files = sorted(glob.glob(os.path.join(JOMARK_LBL, "*.txt")))
    random.shuffle(lbl_files)
    for lf in lbl_files:
        base = os.path.splitext(os.path.basename(lf))[0]
        ip   = os.path.join(JOMARK_IMG, base + ".png")
        if not os.path.exists(ip):
            ip = os.path.join(JOMARK_IMG, base + ".jpg")
        if not os.path.exists(ip):
            continue
        try:
            img = cv2.imread(ip)
            if img is None:
                continue
            ih2, iw2 = img.shape[:2]
            for line in open(lf):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if cid not in JOMARKOW_MAP:
                    continue
                if cc_j[cid] >= MAX_J:
                    continue
                cls = JOMARKOW_MAP[cid]
                cx, cy, bw, bh = (
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])
                )
                x1 = max(0,   int((cx - bw / 2) * iw2))
                y1 = max(0,   int((cy - bh / 2) * ih2))
                x2 = min(iw2, int((cx + bw / 2) * iw2))
                y2 = min(ih2, int((cy + bh / 2) * ih2))
                if x2 > x1 + 8 and y2 > y1 + 8:
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        crops_rgb.append(
                            cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2RGB)
                        )
                        labels_raw.append(cls)
                        cc_j[cid] += 1
        except Exception:
            pass
    print(f"  Jomarkow: added {sum(cc_j.values())} crops | "
          f"{{'helmet': {cc_j[0]}, 'no_ppe': {cc_j[1]}}}")

    crops_rgb  = np.array(crops_rgb)
    labels_raw = np.array(labels_raw)
    np.save(CACHE_X, crops_rgb)
    np.save(CACHE_Y, labels_raw)
    print(f"  Total: {len(crops_rgb)} crops | parsed in {time.time() - t0:.1f}s")

class_counts = dict(zip(*np.unique(labels_raw, return_counts=True)))
print(f"  Class distribution: {class_counts}")


# ══════════════════════════════════════════════════════════════════
# 2.  PREPARE TENSORS + DATALOADERS
# ══════════════════════════════════════════════════════════════════
print(f"\n[2/4] Preparing data tensors ...")

le_cnn = LabelEncoder()
y_cnn  = le_cnn.fit_transform(labels_raw)

# Normalise to ImageNet mean/std (same as production)
X_arr = crops_rgb.astype(np.float32) / 255.0
mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
X_arr = (X_arr - mean) / std
X_t   = torch.tensor(X_arr.transpose(0, 3, 1, 2))   # NCHW
y_t   = torch.tensor(y_cnn, dtype=torch.long)

tr_idx, te_idx = train_test_split(
    range(len(y_cnn)), test_size=0.2, random_state=42, stratify=y_cnn
)

pin = DEVICE.type == 'cuda'
tr_dl = DataLoader(
    TensorDataset(X_t[tr_idx], y_t[tr_idx]),
    batch_size=BATCH, shuffle=True,  num_workers=num_workers, pin_memory=pin
)
te_dl = DataLoader(
    TensorDataset(X_t[te_idx], y_t[te_idx]),
    batch_size=BATCH, shuffle=False, num_workers=num_workers, pin_memory=pin
)
print(f"  Train={len(tr_idx)}, Val={len(te_idx)}, Batches/epoch={len(tr_dl)}")
print(f"  Classes: {list(le_cnn.classes_)}")


# ══════════════════════════════════════════════════════════════════
# 3.  MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════
class PPENet(nn.Module):
    """Lightweight CNN optimised for 64x64 PPE crop classification"""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64→32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 2: 32→16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 3: 16→8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Global average pool → 128 features
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


# ══════════════════════════════════════════════════════════════════
# 4.  EARLY STOPPING CLASS
# ══════════════════════════════════════════════════════════════════
class EarlyStopping:
    """
    Tracks best validation accuracy.  Stops when accuracy has not improved
    for `patience` consecutive epochs.  Saves a CPU copy of the best weights.
    """
    def __init__(self, patience: int = 10):
        self.patience   = patience
        self.best_acc   = 0.0
        self.counter    = 0
        self.best_state = None
        self.stopped    = False

    def step(self, val_acc: float, model: nn.Module) -> bool:
        """
        Call once per epoch.
        Returns True when training should stop.
        """
        if val_acc > self.best_acc:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
        return False


# ══════════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════
print(f"\n[3/4] Training PPENet with early stopping (patience={PATIENCE}) ...")

model     = PPENet(len(le_cnn.classes_)).to(DEVICE)
n_params  = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Construct scheduler over the full budget; early stopping truncates the loop.
# steps_per_epoch and epochs are fixed to 100 so the LR schedule shape is
# identical to production regardless of when we actually stop.
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=MAX_EPOCHS,
    steps_per_epoch=len(tr_dl),
    pct_start=0.2,
    div_factor=10,
    final_div_factor=100,
)

es = EarlyStopping(patience=PATIENCE)
history = {
    'tr_loss': [], 'va_loss': [],
    'tr_acc':  [], 'va_acc':  [],
    'lr':      [],
}

t_train     = time.time()
stopped_epoch = MAX_EPOCHS   # will be overwritten if early stopping fires

for epoch in range(1, MAX_EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────
    model.train()
    tl = tc = tt = 0
    for imgs, lbls in tr_dl:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        out  = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        tl += loss.item() * len(lbls)
        tc += (out.argmax(1) == lbls).sum().item()
        tt += len(lbls)

    # ── Validate ───────────────────────────────────────────────
    model.eval()
    vl = vc = vt = 0
    with torch.no_grad():
        for imgs, lbls in te_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            vl += criterion(out, lbls).item() * len(lbls)
            vc += (out.argmax(1) == lbls).sum().item()
            vt += len(lbls)

    ta = tc / tt
    va = vc / vt
    history['tr_loss'].append(tl / tt)
    history['va_loss'].append(vl / vt)
    history['tr_acc'].append(ta)
    history['va_acc'].append(va)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    if epoch % 5 == 0 or epoch == 1:
        elapsed = time.time() - t_train
        eta     = elapsed / epoch * (MAX_EPOCHS - epoch)
        print(
            f"  Ep {epoch:>3}/{MAX_EPOCHS}  tr={ta:.3f}  val={va:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.5f}  "
            f"patience={es.counter}/{PATIENCE}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
        )

    # ── Early stopping check ───────────────────────────────────
    if es.step(va, model):
        stopped_epoch = epoch
        print(f"\n  Early stopping at epoch {stopped_epoch}  "
              f"(best val acc = {es.best_acc:.4f})")
        break
else:
    # Completed all epochs without early stopping — capture best state
    stopped_epoch = MAX_EPOCHS
    if es.best_state is None:
        # Edge case: first epoch was already the best
        es.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

train_time = time.time() - t_train
print(f"\n  Training done: {train_time:.1f}s | stopped_epoch={stopped_epoch} | "
      f"best_val_acc={es.best_acc:.4f}")


# ══════════════════════════════════════════════════════════════════
# 6.  EVALUATE BEST MODEL
# ══════════════════════════════════════════════════════════════════
print(f"\n[4/4] Evaluating best model & saving outputs ...")

model.load_state_dict(es.best_state)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in te_dl:
        imgs = imgs.to(DEVICE)
        out  = model(imgs)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(lbls.numpy())

rpt = classification_report(
    all_labels, all_preds,
    target_names=le_cnn.classes_, output_dict=True
)
cm  = confusion_matrix(all_labels, all_preds)

print("\n  Classification Report:")
print(classification_report(all_labels, all_preds, target_names=le_cnn.classes_))

accuracy    = rpt['accuracy']
macro_f1    = rpt['macro avg']['f1-score']
weighted_f1 = rpt['weighted avg']['f1-score']


# ── 6a. Save model weights ─────────────────────────────────────
model_path = os.path.join(OUT_DIR, "es_cnn_model.pth")
torch.save(
    {
        'state_dict':    es.best_state,
        'classes':       list(le_cnn.classes_),
        'stopped_epoch': stopped_epoch,
        'best_val_acc':  es.best_acc,
        'arch':          'PPENet-ES',
    },
    model_path
)
print(f"  Saved: {model_path}")


# ── 6b. Training curves with vertical stopped_epoch marker ────
ep_r = range(1, len(history['tr_loss']) + 1)   # actual epochs run

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
ax = axes[0]
ax.plot(ep_r, history['tr_loss'], 'r-', lw=2, label='Train')
ax.plot(ep_r, history['va_loss'], 'b-', lw=2, label='Val')
ax.fill_between(ep_r, history['tr_loss'], alpha=0.1, color='r')
ax.fill_between(ep_r, history['va_loss'], alpha=0.1, color='b')
ax.axvline(x=stopped_epoch, color='gray', lw=1.5, ls='--',
           label=f'Stopped @ {stopped_epoch}')
ax.set_title("Training Loss")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

# Accuracy
ax = axes[1]
ax.plot(ep_r, history['tr_acc'], 'r-', lw=2, label='Train')
ax.plot(ep_r, history['va_acc'], 'b-', lw=2, label='Val')
ax.fill_between(ep_r, history['tr_acc'], alpha=0.1, color='r')
ax.fill_between(ep_r, history['va_acc'], alpha=0.1, color='b')
ax.axhline(es.best_acc, color='g', lw=1.5, ls='--',
           label=f'Best val={es.best_acc:.3f}')
ax.axvline(x=stopped_epoch, color='gray', lw=1.5, ls='--',
           label=f'Stopped @ {stopped_epoch}')
ax.set_title("Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)

# Learning rate
ax = axes[2]
ax.plot(ep_r, history['lr'], color='purple', lw=2)
ax.axvline(x=stopped_epoch, color='gray', lw=1.5, ls='--',
           label=f'Stopped @ {stopped_epoch}')
ax.set_title("Learning Rate (OneCycleLR)")
ax.set_xlabel("Epoch")
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle(
    f"PPENet Early Stopping — stopped at epoch {stopped_epoch}/{MAX_EPOCHS}  "
    f"(patience={PATIENCE}, best_val={es.best_acc:.3f})\n"
    f"{len(crops_rgb)} crops, {len(tr_idx)} train / {len(te_idx)} val",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
training_plot = os.path.join(OUT_DIR, "es_cnn_training.png")
plt.savefig(training_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {training_plot}")


# ── 6c. Confusion matrix ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=le_cnn.classes_).plot(
    ax=ax, cmap='Blues', colorbar=False, values_format='d'
)
ax.set_title(
    f"PPENet-ES Confusion Matrix\n"
    f"acc={accuracy:.3f}  stopped_epoch={stopped_epoch}"
)
ax.tick_params(axis='x', rotation=35)
plt.tight_layout()
confusion_plot = os.path.join(OUT_DIR, "es_cnn_confusion.png")
plt.savefig(confusion_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {confusion_plot}")


# ── 6d. Results CSV ────────────────────────────────────────────
results_df = pd.DataFrame([{
    'Model':           'PPENet-ES',
    'Task':            'multi-class (5)',
    'Accuracy':        round(accuracy,    4),
    'Macro_F1':        round(macro_f1,    4),
    'Weighted_F1':     round(weighted_f1, 4),
    'Stopped_Epoch':   stopped_epoch,
    'Train_Time(s)':   round(train_time,  1),
}])
csv_path = os.path.join(OUT_DIR, "es_cnn_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

print("\n" + "=" * 65)
print("EARLY STOPPING TRAINING COMPLETE")
print("=" * 65)
print(f"  Model        : PPENet-ES")
print(f"  Stopped epoch: {stopped_epoch} / {MAX_EPOCHS}")
print(f"  Best val acc : {es.best_acc:.4f}")
print(f"  Test accuracy: {accuracy:.4f}")
print(f"  Macro F1     : {macro_f1:.4f}")
print(f"  Weighted F1  : {weighted_f1:.4f}")
print(f"  Train time   : {train_time:.1f}s")
print("=" * 65)
