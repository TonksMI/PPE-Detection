"""
PPE RNN TRAINING SCRIPT
========================
Workspace Safety Equipment Detection — RNN Sequence Classifiers
Version 1.0

Architecture:
  - Frozen PPENet encoder (prod_cnn_model.pth) → 128-dim features per crop
  - Synthetic sequences: each crop is augmented seq_len=8 times → RNN aggregation
  - Two models trained:
      FastGRU  : GRU(128→128, 1 layer)  + head
      NormalLSTM: LSTM(128→256, 2 layers) + head

Outputs (all to results/models/):
  rnn_gru_model.pth       — GRU best weights + metadata
  rnn_lstm_model.pth      — LSTM best weights + metadata
  rnn_training_curves.png — 2×2 grid (GRU loss, GRU acc, LSTM loss, LSTM acc)
  rnn_confusion.png       — side-by-side confusion matrices
  rnn_results.csv         — two-row metrics summary

Usage:
  python src/ppe_rnn_train.py
"""

import os
import sys
import random
import time
import warnings

import matplotlib
matplotlib.use('Agg')  # MUST be before any other matplotlib/pyplot import
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Device ──────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Path resolution (Windows / Linux auto-detect) ──────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)   # PPE-Detection/
BASE        = os.path.dirname(PROJECT_DIR)  # D:\Claude or /sessions/...

if os.path.exists("D:/datasets/jomarkow"):
    DATASETS = "D:/datasets"
else:
    DATASETS = os.path.join(BASE, "datasets")

OUT_DIR   = os.path.join(PROJECT_DIR, "results", "models")
CACHE_DIR = os.path.join(BASE, "cache")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ALL_CLASSES = ["full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]
num_workers = 0   # Windows DataLoader constraint

SEQ_LEN = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════════
# 1.  GUARD: prod_cnn_model.pth must exist
# ══════════════════════════════════════════════════════════════════════════════════
CNN_MODEL_PATH = os.path.join(OUT_DIR, "prod_cnn_model.pth")
if not os.path.exists(CNN_MODEL_PATH):
    print(
        f"[ERROR] prod_cnn_model.pth not found at:\n  {CNN_MODEL_PATH}\n"
        "Run src/ppe_production_train.py first to train and save the CNN model."
    )
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════════
# 2.  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════════
class PPENet(nn.Module):
    """Lightweight CNN for 64×64 PPE crop classification (exact production copy)."""
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


class PPERNNClassifier(nn.Module):
    """
    Wraps a frozen PPENet encoder with a trainable RNN head.

    forward input:  (batch, seq_len, 3, 64, 64)
    forward output: (batch, num_classes)
    """
    def __init__(self, encoder, rnn, head):
        super().__init__()
        # Store encoder without registering as a proper submodule so its
        # parameters are excluded from the optimizer parameter group.
        # We assign directly to __dict__ to bypass nn.Module's __setattr__.
        object.__setattr__(self, 'encoder', encoder)
        self.rnn  = rnn
        self.head = head

    def forward(self, x):
        batch, seq_len = x.shape[:2]
        # Flatten batch and sequence dims for the encoder
        x_flat = x.view(batch * seq_len, 3, x.shape[3], x.shape[4])
        # Extract spatial features (batch*seq_len, 128, 1, 1)
        feats = self.encoder.features(x_flat)
        feats = feats.view(batch * seq_len, 128)
        feats = feats.view(batch, seq_len, 128)
        # RNN over the feature sequence
        rnn_out, _ = self.rnn(feats)     # (batch, seq_len, hidden)
        last = rnn_out[:, -1, :]          # (batch, hidden)
        return self.head(last)            # (batch, num_classes)


# ══════════════════════════════════════════════════════════════════════════════════
# 3.  ENCODER LOADING
# ══════════════════════════════════════════════════════════════════════════════════
def load_ppenet_encoder(model_path, device):
    """
    Load prod_cnn_model.pth, freeze all parameters, set eval mode.
    Returns the frozen PPENet model.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    num_classes = len(checkpoint.get('classes', ALL_CLASSES))
    model = PPENet(num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    return model


# ══════════════════════════════════════════════════════════════════════════════════
# 4.  DATASET
# ══════════════════════════════════════════════════════════════════════════════════
class AugSequenceDataset(Dataset):
    """
    Synthetic sequence dataset: each crop is augmented seq_len independent ways.
    Returns (seq_len, 3, 64, 64) float tensor + int label.
    """
    def __init__(self, crops_rgb, labels_int, seq_len=8, train=True):
        self.crops  = crops_rgb      # (N, 64, 64, 3) uint8
        self.labels = labels_int     # (N,) int
        self.seq_len = seq_len
        self.train   = train

        if train:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(20, fill=0),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.RandomErasing(p=0.3),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.crops[idx])   # uint8 numpy → PIL
        views = []
        for _ in range(self.seq_len):
            views.append(self.transform(img))    # (3, 64, 64) float
        seq = torch.stack(views, dim=0)          # (seq_len, 3, 64, 64)
        return seq, int(self.labels[idx])


# ══════════════════════════════════════════════════════════════════════════════════
# 5.  EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════════
class EarlyStopping:
    """
    Tracks best validation accuracy.
    Stops when accuracy has not strictly improved for `patience` consecutive epochs.
    """
    def __init__(self, patience: int = 10):
        self.patience   = patience
        self.best_acc   = -1.0
        self.counter    = 0
        self.best_state = None

    def step(self, val_acc: float, state_dict: dict) -> bool:
        """Returns True when training should stop."""
        if val_acc > self.best_acc:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in state_dict.items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════════════
# 6.  SHARED TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════════
def train_rnn(model, tr_dl, va_dl, config, name):
    """
    Train one RNN model for the given config.

    Returns:
        history       — dict with lists: tr_loss, va_loss, tr_acc, va_acc
        best_state    — CPU state_dict at best val acc
        stopped_epoch — epoch at which training ended
        best_val_acc  — best validation accuracy achieved
        train_time    — wall-clock seconds
    """
    epochs    = config['epochs']
    patience  = config['patience']
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Only optimise parameters that require gradients (RNN + head, not encoder)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    es = EarlyStopping(patience=patience)
    history = {'tr_loss': [], 'va_loss': [], 'tr_acc': [], 'va_acc': []}

    t0 = time.time()
    stopped_epoch = epochs

    print(f"\n  {'='*60}")
    print(f"  Training {name}")
    print(f"  Epochs={epochs}  LR={config['lr']}  Batch={config['batch_size']}  "
          f"Patience={patience}")
    print(f"  {'='*60}")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        tl = tc = tt = 0
        for seqs, lbls in tr_dl:
            seqs = seqs.to(DEVICE)
            lbls = lbls.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seqs)
            loss   = criterion(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            tl += loss.item() * len(lbls)
            tc += (logits.argmax(1) == lbls).sum().item()
            tt += len(lbls)

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        vl = vc = vt = 0
        with torch.no_grad():
            for seqs, lbls in va_dl:
                seqs = seqs.to(DEVICE)
                lbls = lbls.to(DEVICE)
                logits = model(seqs)
                vl += criterion(logits, lbls).item() * len(lbls)
                vc += (logits.argmax(1) == lbls).sum().item()
                vt += len(lbls)

        scheduler.step()

        ta = tc / tt
        va = vc / vt
        history['tr_loss'].append(tl / tt)
        history['va_loss'].append(vl / vt)
        history['tr_acc'].append(ta)
        history['va_acc'].append(va)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            eta = elapsed / epoch * (epochs - epoch)
            print(
                f"  Ep {epoch:>3}/{epochs}  tr={ta:.3f}  val={va:.3f}  "
                f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                f"patience={es.counter}/{patience}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

        # ── Early stopping ─────────────────────────────────────────
        # Collect only the trainable submodule state dicts (rnn + head)
        trainable_state = {}
        for k, v in model.rnn.state_dict().items():
            trainable_state[f'rnn.{k}'] = v
        for k, v in model.head.state_dict().items():
            trainable_state[f'head.{k}'] = v

        if es.step(va, trainable_state):
            stopped_epoch = epoch
            print(f"\n  Early stopping at epoch {stopped_epoch}  "
                  f"(best val acc = {es.best_acc:.4f})")
            break
    else:
        stopped_epoch = epochs
        # Ensure best_state is captured even if training never improved past epoch 1
        if es.best_state is None:
            trainable_state = {}
            for k, v in model.rnn.state_dict().items():
                trainable_state[f'rnn.{k}'] = v
            for k, v in model.head.state_dict().items():
                trainable_state[f'head.{k}'] = v
            es.best_state = {k: v.cpu().clone() for k, v in trainable_state.items()}

    train_time = time.time() - t0
    print(f"\n  {name} done: {train_time:.1f}s | stopped_epoch={stopped_epoch} | "
          f"best_val_acc={es.best_acc:.4f}")

    return history, es.best_state, stopped_epoch, es.best_acc, train_time


# ══════════════════════════════════════════════════════════════════════════════════
# 7.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════════
CACHE_X = os.path.join(CACHE_DIR, "crops_X_600.npy")
CACHE_Y = os.path.join(CACHE_DIR, "crops_y_600.npy")

if not os.path.exists(CACHE_X) or not os.path.exists(CACHE_Y):
    print(
        f"[ERROR] Cache files not found:\n  {CACHE_X}\n  {CACHE_Y}\n"
        "Run src/ppe_production_train.py first to build the crop cache."
    )
    sys.exit(1)

print("=" * 65)
print("PPE RNN TRAINING — FastGRU + NormalLSTM")
print(f"Device : {DEVICE}")
print(f"Seq len: {SEQ_LEN}")
print("=" * 65)

print(f"\n[1/5] Loading cached crops from {CACHE_DIR} ...")
crops_rgb  = np.load(CACHE_X)    # (N, 64, 64, 3) uint8
labels_raw = np.load(CACHE_Y)    # (N,) str
print(f"  Loaded {len(crops_rgb)} crops  shape={crops_rgb.shape}")

# Encode labels
le = LabelEncoder()
y_int = le.fit_transform(labels_raw)
print(f"  Classes: {list(le.classes_)}")
dist = dict(zip(*np.unique(labels_raw, return_counts=True)))
print(f"  Class distribution: {dist}")

# Train / val split
tr_idx, te_idx = train_test_split(
    range(len(y_int)), test_size=0.2, random_state=42, stratify=y_int
)
print(f"  Train={len(tr_idx)}, Val={len(te_idx)}")

print(f"\n[2/5] Building AugSequenceDatasets (seq_len={SEQ_LEN}) ...")
tr_ds = AugSequenceDataset(crops_rgb[tr_idx], y_int[tr_idx], seq_len=SEQ_LEN, train=True)
va_ds = AugSequenceDataset(crops_rgb[te_idx], y_int[te_idx], seq_len=SEQ_LEN, train=False)
print(f"  Train dataset: {len(tr_ds)} samples")
print(f"  Val   dataset: {len(va_ds)} samples")


# ══════════════════════════════════════════════════════════════════════════════════
# 8.  LOAD FROZEN ENCODER
# ══════════════════════════════════════════════════════════════════════════════════
print(f"\n[3/5] Loading frozen PPENet encoder from {CNN_MODEL_PATH} ...")
encoder = load_ppenet_encoder(CNN_MODEL_PATH, DEVICE)
print(f"  Encoder loaded and frozen — {sum(p.numel() for p in encoder.parameters()):,} params")

num_classes = len(le.classes_)


# ══════════════════════════════════════════════════════════════════════════════════
# 9.  BUILD MODELS + TRAIN
# ══════════════════════════════════════════════════════════════════════════════════
print(f"\n[4/5] Training RNN classifiers ...")

# ── FastGRU ────────────────────────────────────────────────────────────────────
gru_config = {
    'epochs':      50,
    'batch_size':  64,
    'lr':          3e-4,
    'weight_decay': 1e-4,
    'patience':    10,
}

gru_rnn  = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
gru_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))
gru_model = PPERNNClassifier(encoder, gru_rnn, gru_head).to(DEVICE)

gru_tr_dl = DataLoader(tr_ds, batch_size=gru_config['batch_size'],
                       shuffle=True,  num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
gru_va_dl = DataLoader(va_ds, batch_size=gru_config['batch_size'],
                       shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))

gru_history, gru_best_state, gru_stopped, gru_best_acc, gru_time = train_rnn(
    gru_model, gru_tr_dl, gru_va_dl, gru_config, "FastGRU"
)

# ── NormalLSTM ─────────────────────────────────────────────────────────────────
lstm_config = {
    'epochs':      50,
    'batch_size':  32,
    'lr':          1e-4,
    'weight_decay': 1e-4,
    'patience':    10,
}

lstm_rnn  = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
                    batch_first=True, dropout=0.3)
lstm_head = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))
lstm_model = PPERNNClassifier(encoder, lstm_rnn, lstm_head).to(DEVICE)

lstm_tr_dl = DataLoader(tr_ds, batch_size=lstm_config['batch_size'],
                        shuffle=True,  num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
lstm_va_dl = DataLoader(va_ds, batch_size=lstm_config['batch_size'],
                        shuffle=False, num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))

lstm_history, lstm_best_state, lstm_stopped, lstm_best_acc, lstm_time = train_rnn(
    lstm_model, lstm_tr_dl, lstm_va_dl, lstm_config, "NormalLSTM"
)


# ══════════════════════════════════════════════════════════════════════════════════
# 10. EVALUATE BEST MODELS + SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════════
print(f"\n[5/5] Evaluating best models & saving outputs ...")


def evaluate_model(model, best_state, val_dl, class_names):
    """
    Load best_state into model (rnn + head only), run inference on val_dl.
    Returns (all_preds, all_labels, report_dict, cm).
    """
    # Restore RNN + head weights from best_state
    rnn_sd  = {k[len('rnn.'):]:  v for k, v in best_state.items() if k.startswith('rnn.')}
    head_sd = {k[len('head.'): ]: v for k, v in best_state.items() if k.startswith('head.')}
    model.rnn.load_state_dict(rnn_sd)
    model.head.load_state_dict(head_sd)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, lbls in val_dl:
            seqs = seqs.to(DEVICE)
            logits = model(seqs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    rpt = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds)
    return all_preds, all_labels, rpt, cm


class_names = list(le.classes_)

gru_preds,  gru_labels,  gru_rpt,  gru_cm  = evaluate_model(
    gru_model,  gru_best_state,  gru_va_dl,  class_names
)
lstm_preds, lstm_labels, lstm_rpt, lstm_cm = evaluate_model(
    lstm_model, lstm_best_state, lstm_va_dl, class_names
)

print("\n  FastGRU Classification Report:")
print(classification_report(gru_labels,  gru_preds,  target_names=class_names))
print("  NormalLSTM Classification Report:")
print(classification_report(lstm_labels, lstm_preds, target_names=class_names))


# ── 10a. Save model weights ───────────────────────────────────────────────────
gru_save_path = os.path.join(OUT_DIR, "rnn_gru_model.pth")
torch.save(
    {
        'state_dict':    gru_best_state,
        'classes':       class_names,
        'arch':          'FastGRU',
        'best_val_acc':  gru_best_acc,
        'stopped_epoch': gru_stopped,
    },
    gru_save_path
)
print(f"\n  Saved: {gru_save_path}")

lstm_save_path = os.path.join(OUT_DIR, "rnn_lstm_model.pth")
torch.save(
    {
        'state_dict':    lstm_best_state,
        'classes':       class_names,
        'arch':          'NormalLSTM',
        'best_val_acc':  lstm_best_acc,
        'stopped_epoch': lstm_stopped,
    },
    lstm_save_path
)
print(f"  Saved: {lstm_save_path}")


# ── 10b. Training curves — 2×2 grid ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def _plot_curves(ax_loss, ax_acc, history, name, stopped_epoch, best_acc):
    ep_r = range(1, len(history['tr_loss']) + 1)

    ax_loss.plot(ep_r, history['tr_loss'], 'r-', lw=2, label='Train')
    ax_loss.plot(ep_r, history['va_loss'], 'b-', lw=2, label='Val')
    ax_loss.fill_between(ep_r, history['tr_loss'], alpha=0.1, color='r')
    ax_loss.fill_between(ep_r, history['va_loss'], alpha=0.1, color='b')
    ax_loss.axvline(x=stopped_epoch, color='gray', lw=1.5, ls='--',
                    label=f'Stopped @ {stopped_epoch}')
    ax_loss.set_title(f"{name} — Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(ep_r, history['tr_acc'], 'r-', lw=2, label='Train')
    ax_acc.plot(ep_r, history['va_acc'], 'b-', lw=2, label='Val')
    ax_acc.fill_between(ep_r, history['tr_acc'], alpha=0.1, color='r')
    ax_acc.fill_between(ep_r, history['va_acc'], alpha=0.1, color='b')
    ax_acc.axhline(best_acc, color='g', lw=1.5, ls='--',
                   label=f'Best val={best_acc:.3f}')
    ax_acc.axvline(x=stopped_epoch, color='gray', lw=1.5, ls='--',
                   label=f'Stopped @ {stopped_epoch}')
    ax_acc.set_title(f"{name} — Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

_plot_curves(axes[0, 0], axes[0, 1], gru_history,  "FastGRU",    gru_stopped,  gru_best_acc)
_plot_curves(axes[1, 0], axes[1, 1], lstm_history, "NormalLSTM", lstm_stopped, lstm_best_acc)

plt.suptitle(
    "PPE RNN Training Curves — FastGRU vs NormalLSTM\n"
    f"(Frozen PPENet encoder, seq_len={SEQ_LEN}, synthetic augmentation sequences)",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
curves_path = os.path.join(OUT_DIR, "rnn_training_curves.png")
plt.savefig(curves_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {curves_path}")


# ── 10c. Side-by-side confusion matrices ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay(gru_cm, display_labels=class_names).plot(
    ax=ax1, cmap='Blues', colorbar=False, values_format='d'
)
ax1.set_title(
    f"FastGRU\nacc={gru_rpt['accuracy']:.3f}  stopped_epoch={gru_stopped}"
)
ax1.tick_params(axis='x', rotation=35)

ConfusionMatrixDisplay(lstm_cm, display_labels=class_names).plot(
    ax=ax2, cmap='Oranges', colorbar=False, values_format='d'
)
ax2.set_title(
    f"NormalLSTM\nacc={lstm_rpt['accuracy']:.3f}  stopped_epoch={lstm_stopped}"
)
ax2.tick_params(axis='x', rotation=35)

plt.suptitle(
    "PPE RNN Confusion Matrices\n"
    f"(Frozen PPENet encoder, seq_len={SEQ_LEN})",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
confusion_path = os.path.join(OUT_DIR, "rnn_confusion.png")
plt.savefig(confusion_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {confusion_path}")


# ── 10d. Results CSV ──────────────────────────────────────────────────────────
results_df = pd.DataFrame([
    {
        'Model':          'FastGRU',
        'Task':           'multi-class (5)',
        'Accuracy':       round(gru_rpt['accuracy'],                   4),
        'Macro_F1':       round(gru_rpt['macro avg']['f1-score'],      4),
        'Weighted_F1':    round(gru_rpt['weighted avg']['f1-score'],   4),
        'Architecture':   'GRU(128→128, 1 layer)',
        'Stopped_Epoch':  gru_stopped,
        'Train_Time(s)':  round(gru_time, 1),
    },
    {
        'Model':          'NormalLSTM',
        'Task':           'multi-class (5)',
        'Accuracy':       round(lstm_rpt['accuracy'],                  4),
        'Macro_F1':       round(lstm_rpt['macro avg']['f1-score'],     4),
        'Weighted_F1':    round(lstm_rpt['weighted avg']['f1-score'],  4),
        'Architecture':   'LSTM(128→256, 2 layers)',
        'Stopped_Epoch':  lstm_stopped,
        'Train_Time(s)':  round(lstm_time, 1),
    },
])
csv_path = os.path.join(OUT_DIR, "rnn_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════════
# 11. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("RNN TRAINING COMPLETE")
print("=" * 65)
for _, row in results_df.iterrows():
    print(f"  {row['Model']:<14} | acc={row['Accuracy']:.4f}  "
          f"macro_F1={row['Macro_F1']:.4f}  "
          f"stopped_ep={row['Stopped_Epoch']}  "
          f"time={row['Train_Time(s)']}s")
print("=" * 65)
