"""
PPE UNET SEGMENTATION TRAINING
================================
Task 4: Train a UNet for pixel-level 5-class PPE segmentation using
SAM-generated pseudo-masks from ppe_mask_generator.py.

Architecture: Standard UNet (~7.7M params)
  Encoder: 4 stages (32→64→128→256 filters) + MaxPool2d
  Bottleneck: 512 filters
  Decoder: 4 stages with skip connections (ConvTranspose2d + DoubleConv)
  Head: Conv2d(32→5, 1×1)

Input:  (B, 3, 256, 256) — RGB, ImageNet normalised
Output: (B, 5, 256, 256) — per-pixel class logits

Outputs (all to results/models/):
  unet_model.pth       — best weights + metadata
  unet_training.png    — loss + mIoU curves with early-stop marker
  unet_predictions.png — 6-image grid (original | GT mask | pred mask)
  unet_results.csv     — single-row metrics summary

Requires: results/models/mask_index.csv (run ppe_mask_generator.py first)

Usage:
  python src/ppe_unet_train.py
"""

import os
import sys
import csv
import json
import random
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Path resolution (Windows / Linux auto-detect) ──────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)   # PPE-Detection/
BASE        = os.path.dirname(PROJECT_DIR)  # D:\Claude or /sessions/...

if os.path.exists("D:/datasets/jomarkow"):
    DATASETS = "D:/datasets"
else:
    DATASETS = os.path.join(BASE, "datasets")

OUT_DIR = os.path.join(PROJECT_DIR, "results", "models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_CLASSES = ["full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]
NUM_CLASSES = len(ALL_CLASSES)

# Class index: full_ppe=0, helmet=1, no_ppe=2, partial_ppe=3, safety_vest=4
MASK_COLORS = {
    0:   (128,   0, 128),  # full_ppe    → purple
    1:   ( 39, 174,  96),  # helmet      → green
    2:   (231,  76,  60),  # no_ppe      → red
    3:   (230, 126,  34),  # partial_ppe → orange
    4:   ( 41, 128, 185),  # safety_vest → blue
    255: (  0,   0,   0),  # background  → black
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE    = 256
EPOCHS      = 50
BATCH       = 8
LR          = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE    = 10
IGNORE_IDX  = 255

num_workers = 0
pin_memory  = (DEVICE.type == 'cuda')


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ARCHITECTURE — UNet
# ══════════════════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    """Two (Conv2d → BatchNorm2d → ReLU) blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    Standard UNet for 5-class PPE segmentation.

    Input:  (B, 3,   256, 256)
    Output: (B, 5,   256, 256)  — raw logits
    Params: ~7.7M
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3,    32)
        self.enc2 = DoubleConv(32,   64)
        self.enc3 = DoubleConv(64,  128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4  = DoubleConv(512, 256)

        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3  = DoubleConv(256, 128)

        self.up2   = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)
        self.dec2  = DoubleConv(128,  64)

        self.up1   = nn.ConvTranspose2d( 64,  32, kernel_size=2, stride=2)
        self.dec1  = DoubleConv( 64,  32)

        # Segmentation head
        self.head  = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ───────────────────────────────────────────────
        s1 = self.enc1(x)                    # (B,  32, 256, 256)
        s2 = self.enc2(self.pool(s1))        # (B,  64, 128, 128)
        s3 = self.enc3(self.pool(s2))        # (B, 128,  64,  64)
        s4 = self.enc4(self.pool(s3))        # (B, 256,  32,  32)

        # ── Bottleneck ────────────────────────────────────────────
        b  = self.bottleneck(self.pool(s4))  # (B, 512,  16,  16)

        # ── Decoder ───────────────────────────────────────────────
        d4 = self.dec4(torch.cat([self.up4(b),  s4], dim=1))  # (B, 256, 32,  32)
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))  # (B, 128, 64,  64)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))  # (B,  64, 128, 128)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))  # (B,  32, 256, 256)

        return self.head(d1)                                    # (B,   5, 256, 256)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Monitors validation mIoU; stops when it has not improved for `patience`
    consecutive epochs. Keeps a CPU copy of the best weights.
    """
    def __init__(self, patience: int = 10):
        self.patience    = patience
        self.best_miou   = -1.0
        self.counter     = 0
        self.best_state  = None
        self.stopped     = False

    def step(self, val_miou: float, model: nn.Module) -> bool:
        if val_miou > self.best_miou:
            self.best_miou  = val_miou
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class PPESegDataset(Dataset):
    """
    Loads (image, mask) pairs for UNet segmentation training.

    Args:
        pairs    : list of (img_path, mask_path) tuples
        img_size : spatial size to resize both image and mask to
        train    : if True, applies augmentations
    """
    def __init__(self, pairs, img_size: int = 256, train: bool = True):
        self.pairs    = pairs
        self.img_size = img_size
        self.train    = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # ── Load image ────────────────────────────────────────────
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)

        # ── ColorJitter (image only, before tensorizing) ──────────
        if self.train and random.random() < 0.7:
            jitter    = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
            img_pil   = Image.fromarray(img_rgb)
            img_pil   = jitter(img_pil)
            img_rgb   = np.array(img_pil)

        # ── Normalise and tensorize image ─────────────────────────
        img_f32  = img_rgb.astype(np.float32) / 255.0
        img_f32  = (img_f32 - IMAGENET_MEAN) / IMAGENET_STD          # HWC
        img_t    = torch.from_numpy(img_f32.transpose(2, 0, 1))       # CHW

        # ── Load mask ─────────────────────────────────────────────
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            mask_raw = np.full((self.img_size, self.img_size), IGNORE_IDX, dtype=np.uint8)

        mask_raw = cv2.resize(mask_raw, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)   # CRITICAL: preserve indices
        mask_t   = torch.tensor(mask_raw.astype(np.int64), dtype=torch.long)

        # ── Spatial augmentations (consistent for image + mask) ───
        if self.train:
            # Horizontal flip
            if random.random() < 0.5:
                img_t  = img_t.flip(-1)
                mask_t = mask_t.flip(-1)

            # Random rotation ±10°
            if random.random() < 0.3:
                angle  = random.uniform(-10, 10)
                img_t  = TF.rotate(img_t, angle, fill=0)
                mask_t = TF.rotate(
                    mask_t.unsqueeze(0).float(),
                    angle,
                    interpolation=TF.InterpolationMode.NEAREST,
                    fill=IGNORE_IDX,
                ).squeeze(0).long()

        return img_t, mask_t


# ══════════════════════════════════════════════════════════════════════════════
# 4.  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_miou(pred_masks: torch.Tensor,
                 true_masks: torch.Tensor,
                 num_classes: int = 5,
                 ignore_index: int = 255):
    """
    Compute mean Intersection-over-Union over a batch.

    Args:
        pred_masks  : (N, H, W) int tensor — argmax of logits
        true_masks  : (N, H, W) long tensor
        num_classes : number of foreground classes
        ignore_index: pixel value to exclude from evaluation

    Returns:
        mean_iou      : float — nanmean across valid classes
        per_class_iou : list of num_classes floats (NaN for absent classes)
    """
    pred = pred_masks.view(-1)
    true = true_masks.view(-1)

    # Mask out ignore-index pixels from both pred and true
    valid = (true != ignore_index)
    pred_valid = pred[valid]
    true_valid = true[valid]

    per_class_iou = []
    for c in range(num_classes):
        pred_c = (pred_valid == c)
        true_c = (true_valid == c)
        intersection = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        if union.item() == 0:
            per_class_iou.append(float('nan'))
        else:
            iou = intersection / union
            per_class_iou.append(iou.item())

    valid_ious = [v for v in per_class_iou if not np.isnan(v)]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return mean_iou, per_class_iou


def compute_pixel_acc(pred_masks: torch.Tensor,
                      true_masks: torch.Tensor,
                      ignore_index: int = 255) -> float:
    """Pixel accuracy, excluding ignore_index pixels."""
    valid = (true_masks != ignore_index)
    if valid.sum().item() == 0:
        return 0.0
    correct = ((pred_masks == true_masks) & valid).sum().item()
    return correct / valid.sum().item()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  HELPER — colorize a (H, W) label mask to (H, W, 3) RGB
# ══════════════════════════════════════════════════════════════════════════════

def colorize_mask(mask_np: np.ndarray) -> np.ndarray:
    """Convert a (H, W) uint8/int label mask to a (H, W, 3) RGB image."""
    h, w   = mask_np.shape
    rgb    = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in MASK_COLORS.items():
        rgb[mask_np == idx] = color
    return rgb


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_mask_index(csv_path: str):
    """
    Read mask_index.csv and return list of (img_path, mask_path) tuples
    where both files exist on disk.
    """
    pairs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path  = row.get('img_path',  row.get('image_path', '')).strip()
            mask_path = row.get('mask_path', '').strip()
            if os.path.exists(img_path) and os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ONE EPOCH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n_samples  = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n_samples  += imgs.size(0)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss  = 0.0
    n_samples   = 0
    all_preds   = []
    all_targets = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits      = model(imgs)
        loss        = criterion(logits, masks)
        preds       = logits.argmax(dim=1)
        total_loss += loss.item() * imgs.size(0)
        n_samples  += imgs.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mean_iou, per_class_iou = compute_miou(all_preds, all_targets, NUM_CLASSES, IGNORE_IDX)
    pix_acc = compute_pixel_acc(all_preds, all_targets, IGNORE_IDX)
    return total_loss / max(n_samples, 1), mean_iou, per_class_iou, pix_acc


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

MASK_INDEX = os.path.join(OUT_DIR, "mask_index.csv")
if not os.path.exists(MASK_INDEX):
    print(f"ERROR: {MASK_INDEX} not found. Run ppe_mask_generator.py first.")
    sys.exit(1)

print("=" * 70)
print("PPE UNET — PIXEL-LEVEL SEGMENTATION TRAINING")
print(f"Device : {DEVICE}")
print(f"Epochs : {EPOCHS}  |  Batch : {BATCH}  |  LR : {LR}  |  Patience : {PATIENCE}")
print(f"Image size : {IMG_SIZE}x{IMG_SIZE}  |  Classes : {NUM_CLASSES}")
print("=" * 70)

# ── 8.1  Load data ────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading mask index from {MASK_INDEX} ...")
pairs = load_mask_index(MASK_INDEX)
print(f"  Found {len(pairs)} valid (image, mask) pairs")

if len(pairs) == 0:
    print("ERROR: No valid pairs found. Check that mask_index.csv paths are correct.")
    sys.exit(1)

tr_pairs, va_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
print(f"  Train: {len(tr_pairs)}  |  Val: {len(va_pairs)}")

tr_ds = PPESegDataset(tr_pairs, img_size=IMG_SIZE, train=True)
va_ds = PPESegDataset(va_pairs, img_size=IMG_SIZE, train=False)

tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory)
va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory)

print(f"  Train batches/epoch: {len(tr_dl)}  |  Val batches: {len(va_dl)}")

# ── 8.2  Model ────────────────────────────────────────────────────────────────
print(f"\n[2/5] Building UNet ...")
model    = UNet(num_classes=NUM_CLASSES).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

es = EarlyStopping(patience=PATIENCE)

history = {
    'tr_loss': [],
    'va_loss': [],
    'va_miou': [],
}

# ── 8.3  Training loop ────────────────────────────────────────────────────────
print(f"\n[3/5] Training ({EPOCHS} epochs, early stop patience={PATIENCE}) ...")
t_start       = time.time()
stopped_epoch = EPOCHS

for epoch in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch(model, tr_dl, criterion, optimizer)
    va_loss, va_miou, per_cls, pix_acc = validate(model, va_dl, criterion)
    scheduler.step()

    history['tr_loss'].append(tr_loss)
    history['va_loss'].append(va_loss)
    history['va_miou'].append(va_miou)

    print(f"  Ep {epoch}/{EPOCHS}  tr_loss={tr_loss:.4f}  "
          f"val_loss={va_loss:.4f}  val_mIoU={va_miou:.4f}")

    if es.step(va_miou, model):
        stopped_epoch = epoch
        print(f"\n  Early stopping at epoch {stopped_epoch}  "
              f"(best val mIoU = {es.best_miou:.4f})")
        break
else:
    stopped_epoch = EPOCHS
    if es.best_state is None:
        es.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

train_time = time.time() - t_start
print(f"\n  Training done: {train_time:.1f}s | stopped_epoch={stopped_epoch} | "
      f"best_val_mIoU={es.best_miou:.4f}")

# ── 8.4  Final evaluation with best weights ───────────────────────────────────
print(f"\n[4/5] Final evaluation with best weights ...")
model.load_state_dict(es.best_state)
model.eval()

_, final_miou, final_per_cls, final_pix_acc = validate(model, va_dl, criterion)
print(f"  Final val mIoU  : {final_miou:.4f}")
print(f"  Final pixel acc : {final_pix_acc:.4f}")
for cls_name, iou_val in zip(ALL_CLASSES, final_per_cls):
    nan_str = "N/A" if np.isnan(iou_val) else f"{iou_val:.4f}"
    print(f"    {cls_name:<14}: {nan_str}")

# ── 8.5  Save outputs ─────────────────────────────────────────────────────────
print(f"\n[5/5] Saving outputs ...")

# ── 8.5a  Model weights ───────────────────────────────────────────────────────
per_cls_dict = {
    cls_name: (float(iou_val) if not np.isnan(iou_val) else None)
    for cls_name, iou_val in zip(ALL_CLASSES, final_per_cls)
}
model_path = os.path.join(OUT_DIR, "unet_model.pth")
torch.save(
    {
        'state_dict':    es.best_state,
        'classes':       ALL_CLASSES,
        'best_val_miou': es.best_miou,
        'stopped_epoch': stopped_epoch,
        'arch':          'UNet',
    },
    model_path
)
print(f"  Saved: {model_path}")

# ── 8.5b  Training curves ─────────────────────────────────────────────────────
ep_r = range(1, len(history['tr_loss']) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(ep_r, history['tr_loss'], 'r-', lw=2, label='Train loss')
ax.plot(ep_r, history['va_loss'], 'b-', lw=2, label='Val loss')
ax.fill_between(ep_r, history['tr_loss'], alpha=0.1, color='r')
ax.fill_between(ep_r, history['va_loss'], alpha=0.1, color='b')
if stopped_epoch < EPOCHS:
    ax.axvline(x=stopped_epoch, color='gray', lw=1.5, ls='--',
               label=f'Early stop @ {stopped_epoch}')
ax.set_title("Loss Curves")
ax.set_xlabel("Epoch")
ax.set_ylabel("CrossEntropy Loss")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(ep_r, history['va_miou'], 'g-', lw=2, label='Val mIoU')
ax.fill_between(ep_r, history['va_miou'], alpha=0.15, color='g')
if stopped_epoch < EPOCHS:
    ax.axvline(x=stopped_epoch, color='gray', lw=1.5, ls='--',
               label=f'Early stop @ {stopped_epoch}')
ax.axhline(es.best_miou, color='darkgreen', lw=1.5, ls=':',
           label=f'Best mIoU={es.best_miou:.4f}')
ax.set_title("Validation mIoU")
ax.set_xlabel("Epoch")
ax.set_ylabel("mIoU")
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle(
    f"UNet PPE Segmentation — stopped epoch {stopped_epoch}/{EPOCHS}  "
    f"(best mIoU={es.best_miou:.4f})\n"
    f"Train={len(tr_pairs)}  Val={len(va_pairs)}  Classes={NUM_CLASSES}",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
training_plot = os.path.join(OUT_DIR, "unet_training.png")
plt.savefig(training_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {training_plot}")

# ── 8.5c  Prediction visualisation (6-image grid) ────────────────────────────
# Pick up to 6 validation samples
n_vis  = min(6, len(va_pairs))
vis_idxs = random.sample(range(len(va_pairs)), n_vis)

fig, axes = plt.subplots(n_vis, 3, figsize=(12, 4 * n_vis))
if n_vis == 1:
    axes = axes[np.newaxis, :]  # ensure 2-D indexing

col_titles = ["Original Image", "GT Mask", "Predicted Mask"]
for col_i, title in enumerate(col_titles):
    axes[0, col_i].set_title(title, fontsize=13, fontweight='bold', pad=8)

model.eval()
with torch.no_grad():
    for row_i, vi in enumerate(vis_idxs):
        img_path, mask_path = va_pairs[vi]

        # Load original image for display (un-normalised)
        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            orig_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            orig_rgb = cv2.resize(orig_rgb, (IMG_SIZE, IMG_SIZE))

        # Load GT mask
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            gt_mask = np.full((IMG_SIZE, IMG_SIZE), IGNORE_IDX, dtype=np.uint8)
        else:
            gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_NEAREST)

        # Get model prediction
        # Re-use the dataset's normalisation pipeline
        img_f32 = orig_rgb.astype(np.float32) / 255.0
        img_f32 = (img_f32 - IMAGENET_MEAN) / IMAGENET_STD
        img_t   = torch.from_numpy(img_f32.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        logits  = model(img_t)                        # (1, 5, H, W)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Colorize masks
        gt_colored   = colorize_mask(gt_mask)
        pred_colored = colorize_mask(pred_mask)

        axes[row_i, 0].imshow(orig_rgb)
        axes[row_i, 0].axis('off')

        axes[row_i, 1].imshow(gt_colored)
        axes[row_i, 1].axis('off')

        axes[row_i, 2].imshow(pred_colored)
        axes[row_i, 2].axis('off')

# Legend
legend_patches = [
    plt.Rectangle((0, 0), 1, 1, color=np.array(MASK_COLORS[i]) / 255.0)
    for i in range(NUM_CLASSES)
]
fig.legend(
    legend_patches, ALL_CLASSES,
    loc='lower center', ncol=NUM_CLASSES,
    fontsize=10, framealpha=0.8,
    bbox_to_anchor=(0.5, -0.01),
)
plt.suptitle(
    f"UNet Predictions — val mIoU={final_miou:.4f}  pix_acc={final_pix_acc:.4f}",
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
predictions_plot = os.path.join(OUT_DIR, "unet_predictions.png")
plt.savefig(predictions_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {predictions_plot}")

# ── 8.5d  Results CSV ─────────────────────────────────────────────────────────
per_cls_json = json.dumps({
    cls_name: (round(float(iou_val), 6) if not np.isnan(iou_val) else None)
    for cls_name, iou_val in zip(ALL_CLASSES, final_per_cls)
})

csv_path = os.path.join(OUT_DIR, "unet_results.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'Model', 'Task', 'mIoU', 'Pixel_Acc', 'Per_Class_IoU_JSON', 'Train_Time(s)'
    ])
    writer.writeheader()
    writer.writerow({
        'Model':               'UNet',
        'Task':                'segmentation-5class',
        'mIoU':                round(final_miou,     6),
        'Pixel_Acc':           round(final_pix_acc,  6),
        'Per_Class_IoU_JSON':  per_cls_json,
        'Train_Time(s)':       round(train_time,     1),
    })
print(f"  Saved: {csv_path}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("UNET TRAINING COMPLETE")
print("=" * 70)
print(f"  Architecture  : UNet  ({n_params:,} params)")
print(f"  Stopped epoch : {stopped_epoch} / {EPOCHS}")
print(f"  Best val mIoU : {es.best_miou:.4f}")
print(f"  Final val mIoU: {final_miou:.4f}")
print(f"  Pixel accuracy: {final_pix_acc:.4f}")
print(f"  Train time    : {train_time:.1f}s")
print(f"  Outputs saved to: {OUT_DIR}")
print("=" * 70)
