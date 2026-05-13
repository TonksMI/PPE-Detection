"""
ppe_seg_zeroshot.py
-------------------
Category (c) evaluation: pre-existing architecture with pre-trained weights,
NOT fine-tuned on our data.

Loads the COCO-pretrained DeepLabV3+ ResNet-50 (21 COCO classes) and runs
zero-shot inference on the keremberke val set.  Because the COCO output head
has 21 classes (none of which are our PPE labels), every PPE pixel will be
mis-labelled — yielding near-zero mIoU for PPE classes.  The background class
(index 0) may overlap coincidentally.

This result is deliberately documented to show that pre-trained weights without
task-specific fine-tuning do not transfer to our specialised PPE schema.

Output: results/models/deeplab_zeroshot_results.csv
"""

import os
import sys
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

for _cand in [
    "D:/datasets/ppe_seg_ke",
    "D:/Claude/datasets/ppe_seg_ke",
    str(BASE.parent / "datasets" / "ppe_seg_ke"),
]:
    if os.path.exists(os.path.join(_cand, "semantic", "val", "images")):
        SEG_ROOT = Path(_cand)
        break
else:
    raise FileNotFoundError("ppe_seg_ke dataset not found")

OUT_DIR = BASE / "results" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Our class schema (11 classes) ─────────────────────────────────────────
CLASSES = [
    "background", "helmet", "no_helmet", "glove", "no_glove",
    "goggles", "no_goggles", "mask", "no_mask", "shoes", "no_shoes",
]
N_CLS = len(CLASSES)   # 11
IMG_SZ = 512

# ── Load COCO-pretrained model (21 classes, NOT modified for our data) ─────
print("Loading COCO-pretrained DeepLabV3+ ResNet-50 (21 classes, zero fine-tuning)...")
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
model = model.to(DEVICE).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {n_params / 1e6:.1f}M")
print(f"  Output head: 21 COCO classes  (our schema: {N_CLS} classes)")
print(f"  Fine-tuned : NO — using weights straight from torchvision hub")
print()

# ── Preprocessing (same as training) ─────────────────────────────────────
img_tf = transforms.Compose([
    transforms.Resize((IMG_SZ, IMG_SZ)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def compute_iou(pred, mask, n_classes):
    """Per-class IoU between two integer arrays."""
    per_iou = []
    for c in range(n_classes):
        pred_c = pred == c
        true_c = mask == c
        inter = float((pred_c & true_c).sum())
        union = float((pred_c | true_c).sum())
        per_iou.append(inter / union if union > 0 else float("nan"))
    valid = [v for v in per_iou if not np.isnan(v)]
    mean_iou = sum(valid) / len(valid) if valid else 0.0
    return per_iou, mean_iou


# ── Collect val images ────────────────────────────────────────────────────
img_dir  = SEG_ROOT / "semantic" / "val" / "images"
msk_dir  = SEG_ROOT / "semantic" / "val" / "masks"
img_paths = sorted(img_dir.glob("*.[jp][pn]g"))
print(f"Val images found: {len(img_paths)}")

# ── Evaluate ──────────────────────────────────────────────────────────────
all_per_iou = [[] for _ in range(N_CLS)]
total_correct = 0
total_pixels  = 0
t0 = time.time()

print("Running zero-shot inference ...")
with torch.no_grad():
    for i, img_path in enumerate(img_paths):
        msk_path = msk_dir / (img_path.stem + ".png")
        if not msk_path.exists():
            continue

        img  = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(msk_path).resize((IMG_SZ, IMG_SZ), Image.NEAREST))
        mask[mask >= N_CLS] = 0   # clamp stray indices

        inp  = img_tf(img).unsqueeze(0).to(DEVICE)
        # COCO model outputs 21 classes; our masks use 11 classes.
        # Clamp predictions to [0, N_CLS-1] so the IoU calculation is well-defined.
        # Indices 1-10 in COCO correspond to: aeroplane, bicycle, bird, boat, bottle,
        # bus, car, cat, chair, cow — none are PPE.  Index 0 is background.
        raw_pred = model(inp)["out"].argmax(1).squeeze(0).cpu().numpy()
        pred = np.clip(raw_pred, 0, N_CLS - 1)   # map COCO 11-20 → 10 (no_shoes)

        per_iou, _ = compute_iou(pred, mask, N_CLS)
        for c, v in enumerate(per_iou):
            if not np.isnan(v):
                all_per_iou[c].append(v)

        valid_px = mask < 255
        total_correct += int(((pred == mask) & valid_px).sum())
        total_pixels  += int(valid_px.sum())

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_paths)} images processed  [{time.time()-t0:.0f}s]")

elapsed = time.time() - t0

# ── Aggregate ─────────────────────────────────────────────────────────────
per_iou_mean = [
    float(np.mean(v)) if v else float("nan") for v in all_per_iou
]
valid_class_ious = [v for v in per_iou_mean if not np.isnan(v)]
miou     = float(np.mean(valid_class_ious)) if valid_class_ious else 0.0
pix_acc  = total_correct / total_pixels if total_pixels > 0 else 0.0

# ── Print results ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("DeepLabV3+ ResNet50 — ZERO-SHOT (COCO pretrained, NOT fine-tuned)")
print("=" * 60)
print(f"  mIoU (all classes) : {miou:.4f}  ({miou*100:.2f}%)")
print(f"  Pixel Accuracy     : {pix_acc:.4f}  ({pix_acc*100:.2f}%)")
print(f"  Eval time          : {elapsed:.1f}s")
print()
print(f"  {'Class':<14} {'IoU':>8}")
print(f"  {'-'*24}")
for cls, iou in zip(CLASSES, per_iou_mean):
    s = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
    print(f"  {cls:<14} {s:>8}")
print()
print("NOTE: COCO model output classes (21) do not map to our PPE schema (11).")
print("Background coincidentally shares index 0; all PPE classes score ~0.")
print("This demonstrates that fine-tuning is essential for task transfer.")

# ── Save CSV ───────────────────────────────────────────────────────────────
out_csv = OUT_DIR / "deeplab_zeroshot_results.csv"
fieldnames = (
    ["Model", "Task", "mIoU", "Pixel_Acc", "Train_Time(s)"]
    + [f"IoU_{c}" for c in CLASSES]
)
row = {
    "Model":         "DeepLabV3+ ResNet50 (zero-shot COCO)",
    "Task":          "semantic_seg_zeroshot",
    "mIoU":          round(miou, 4),
    "Pixel_Acc":     round(pix_acc, 4),
    "Train_Time(s)": 0,
}
for c, iou in zip(CLASSES, per_iou_mean):
    row[f"IoU_{c}"] = round(iou, 4) if not np.isnan(iou) else "NaN"

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerow(row)

print(f"\nSaved -> {out_csv}")
