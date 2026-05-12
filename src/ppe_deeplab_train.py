"""
DeepLabV3+ Semantic Segmentation for PPE (6 classes)
=====================================================
Fine-tunes torchvision deeplabv3_resnet50 on the combined segmentation dataset
produced by ppe_seg_data_prep.py.

Classes (pixel values in mask PNG):
  0=background  1=full_ppe  2=helmet  3=no_ppe  4=partial_ppe  5=safety_vest

Outputs (all in results/models/):
  deeplab_model.pth        best weights
  deeplab_results.csv      mIoU, pixel_acc, per-class IoU
  deeplab_training.png     loss + mIoU curves
  deeplab_pred_grid.png    4x4 val prediction grid
  deeplab_confusion.png    per-class IoU heatmap
"""

import os
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────
for _cand in ["D:/datasets/ppe_seg", "D:/Claude/datasets/ppe_seg",
              str(Path(__file__).resolve().parents[2] / "datasets" / "ppe_seg")]:
    if os.path.exists(os.path.join(_cand, "semantic", "train", "images")):
        SEM_ROOT = os.path.join(_cand, "semantic")
        break
else:
    raise FileNotFoundError("ppe_seg/semantic dataset not found — run ppe_seg_data_prep.py first")

OUT_DIR = str(Path(__file__).resolve().parent.parent / "results" / "models")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES   = ["background", "full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]
N_CLASSES = len(CLASSES)   # 6
IGNORE    = 255

# ── Dataset ────────────────────────────────────────────────────────────────
class SegDataset(Dataset):
    SIZE = 512

    def __init__(self, split, augment=False):
        img_dir  = os.path.join(SEM_ROOT, split, "images")
        mask_dir = os.path.join(SEM_ROOT, split, "masks")
        self.samples  = []
        self.augment  = augment

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem      = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, stem + ".png")
            if os.path.exists(mask_path):
                self.samples.append((os.path.join(img_dir, fname), mask_path))

        self.img_tf = transforms.Compose([
            transforms.Resize((self.SIZE, self.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.augment and random.random() < 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img  = self.img_tf(img)
        mask = torch.from_numpy(
            np.array(mask.resize((self.SIZE, self.SIZE),
                                  Image.NEAREST), dtype=np.int64)
        )
        # Clamp invalid class indices to background
        mask = mask.clamp(0, N_CLASSES - 1)
        return img, mask


# ── IoU helpers ────────────────────────────────────────────────────────────
def compute_iou(pred, target, n_classes):
    """Returns per-class IoU (NaN for absent classes) and mean over present."""
    ious = []
    pred   = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    for c in range(n_classes):
        tp = ((pred == c) & (target == c)).sum()
        fp = ((pred == c) & (target != c)).sum()
        fn = ((pred != c) & (target == c)).sum()
        denom = tp + fp + fn
        ious.append(float(tp) / denom if denom > 0 else float('nan'))
    valid = [v for v in ious if not np.isnan(v)]
    return ious, (sum(valid) / len(valid)) if valid else 0.0


# ── Training ───────────────────────────────────────────────────────────────
def main():
    EPOCHS = 30
    BATCH  = 8
    LR     = 3e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 65)
    print("DeepLabV3+ SEMANTIC PPE SEGMENTATION")
    print(f"Device: {DEVICE}  |  Epochs: {EPOCHS}  |  Batch: {BATCH}")
    print("=" * 65)

    train_ds = SegDataset("train", augment=True)
    val_ds   = SegDataset("val",   augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_ldr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                           num_workers=4, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                           num_workers=4, pin_memory=True)

    # Load pretrained DeepLabV3+ and replace classifier head
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(256, N_CLASSES, kernel_size=1)
    model.aux_classifier[-1] = nn.Conv2d(256, N_CLASSES, kernel_size=1)
    model = model.to(DEVICE)

    # Class-frequency weighting (background is ~80% of pixels; upweight rare PPE)
    class_weights = torch.tensor(
        [0.2, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou, best_state = 0.0, None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        tl, n = 0.0, 0
        for imgs, masks in train_ldr:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out["out"], masks) + 0.4 * criterion(out["aux"], masks)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(imgs); n += len(imgs)
        scheduler.step()

        # ── Validate ──
        model.eval()
        vl, vn = 0.0, 0
        all_miou = []
        with torch.no_grad():
            for imgs, masks in val_ldr:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out  = model(imgs)
                loss = criterion(out["out"], masks)
                vl  += loss.item() * len(imgs); vn += len(imgs)
                preds = out["out"].argmax(1)
                for p, m in zip(preds, masks):
                    _, miou = compute_iou(p, m, N_CLASSES)
                    all_miou.append(miou)

        train_loss = tl / n
        val_loss   = vl / vn
        val_miou   = sum(all_miou) / len(all_miou) if all_miou else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        if val_miou > best_miou:
            best_miou  = val_miou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0:
            print(f"Ep {ep:3d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_mIoU={val_miou:.4f}  best={best_miou:.4f}")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f}m  best_mIoU={best_miou:.4f}")

    # ── Save checkpoint ──
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state,
                "classes": CLASSES, "best_miou": best_miou},
               os.path.join(OUT_DIR, "deeplab_model.pth"))

    # ── Final per-class evaluation ──
    model.eval()
    all_preds_flat, all_masks_flat = [], []
    with torch.no_grad():
        for imgs, masks in val_ldr:
            preds = model(imgs.to(DEVICE))["out"].argmax(1)
            all_preds_flat.append(preds.cpu())
            all_masks_flat.append(masks)

    all_preds = torch.cat(all_preds_flat)
    all_masks = torch.cat(all_masks_flat)
    per_class_ious, mean_iou = compute_iou(all_preds, all_masks, N_CLASSES)

    # Pixel accuracy
    valid     = (all_masks != IGNORE)
    pixel_acc = ((all_preds == all_masks) & valid).sum().float() / valid.sum().float()

    print("\nPer-class IoU:")
    for cls, iou in zip(CLASSES, per_class_ious):
        print(f"  {cls:15s}: {iou:.4f}" if not np.isnan(iou) else f"  {cls:15s}: N/A")
    print(f"Mean IoU:  {mean_iou:.4f}")
    print(f"Pixel Acc: {float(pixel_acc):.4f}")

    # ── Write CSV ──
    csv_path = os.path.join(OUT_DIR, "deeplab_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Model", "Task", "mIoU", "Pixel_Acc", "Train_Time(s)"] + \
                 [f"IoU_{c}" for c in CLASSES]
        w.writerow(header)
        row = ["DeepLabV3+ ResNet50", "semantic_seg",
               f"{mean_iou:.4f}", f"{float(pixel_acc):.4f}", f"{elapsed:.1f}"]
        row += [f"{v:.4f}" if not np.isnan(v) else "" for v in per_class_ious]
        w.writerow(row)
    print(f"Saved {csv_path}")

    # ── Training curves ──
    eps = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(eps, history["train_loss"], "r-", label="Train")
    ax1.plot(eps, history["val_loss"],   "b-", label="Val")
    ax1.set_title("DeepLab Loss"); ax1.legend()
    ax2.plot(eps, history["val_miou"], "g-", label="Val mIoU")
    ax2.axhline(best_miou, color="purple", linestyle="--",
                label=f"Best: {best_miou:.3f}")
    ax2.set_title("DeepLab mIoU"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "deeplab_training.png"), dpi=150)
    plt.close()

    # ── Per-class IoU bar chart ──
    present = [(c, v) for c, v in zip(CLASSES, per_class_ious)
               if not np.isnan(v)]
    names, vals = zip(*present) if present else ([], [])
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, vals, color="#4C72B0", edgecolor="white")
    ax.axhline(mean_iou, color="crimson", linestyle="--",
               label=f"Mean IoU={mean_iou:.3f}")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("DeepLabV3+ Per-Class IoU (val)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "deeplab_confusion.png"), dpi=150)
    plt.close()

    # ── Prediction grid (up to 8 samples) ──
    COLORS = np.array([
        [0,   0,   0  ],   # 0 bg      black
        [128, 0,   128],   # 1 full    purple
        [39,  174, 96 ],   # 2 helmet  green
        [231, 76,  60 ],   # 3 no_ppe  red
        [230, 126, 34 ],   # 4 partial orange
        [41,  128, 185],   # 5 vest    blue
    ], dtype=np.uint8)

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    n_samples = min(8, len(val_ds))
    val_items = [val_ds[i] for i in range(n_samples)]
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            img_t, mask_t = val_items[i]
            pred = model(img_t.unsqueeze(0).to(DEVICE))["out"].argmax(1)[0].cpu()
            img_np = ((img_t * std_t + mean_t).permute(1,2,0).numpy() * 255).clip(0,255).astype(np.uint8)
            row, base_col = divmod(i, 2)
            axes[row][base_col*2].imshow(img_np); axes[row][base_col*2].axis("off")
            axes[row][base_col*2].set_title("Image", fontsize=8)
            pred_rgb = COLORS[pred.numpy()]
            axes[row][base_col*2+1].imshow(pred_rgb); axes[row][base_col*2+1].axis("off")
            axes[row][base_col*2+1].set_title("Pred", fontsize=8)

    plt.suptitle("DeepLabV3+ Predictions", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "deeplab_pred_grid.png"), dpi=150)
    plt.close()
    print(f"Saved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
