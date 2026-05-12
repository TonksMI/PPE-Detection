# Pixel-Wise PPE Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two pixel-level PPE detection models — DeepLabV3+ semantic segmentation and YOLOv8n-seg instance segmentation — trained on a combined dataset of existing MinhNKB data (SAM2 pseudo-masks) and the `keremberke/protective-equipment-detection` HuggingFace dataset (SAM2 box-prompted masks).

**Architecture:** A shared data prep script generates two dataset formats from the same source images: 6-class PNG semantic masks (background + 5 PPE classes) for DeepLabV3+, and YOLO polygon format for YOLOv8n-seg. Both models are evaluated independently and added to the project experiment comparison.

**Tech Stack:** PyTorch, torchvision (deeplabv3_resnet50), ultralytics (SAM2, YOLOv8n-seg), huggingface `datasets`, opencv, matplotlib, seaborn

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/ppe_seg_data_prep.py` | Create | Download keremberke HF dataset; generate SAM2 masks for 2000 images; build semantic + YOLO-seg datasets from combined sources |
| `src/ppe_deeplab_train.py` | Create | Fine-tune deeplabv3_resnet50 (6 classes), evaluate per-class IoU, save results |
| `src/ppe_experiment_comparison.py` | Modify | Add loaders for `deeplab_results.csv` and `yolo_seg_results.csv` |
| `reports/generate_assignment_report.py` | Modify | Add pixel-wise section to writeup |
| `results/models/deeplab_model.pth` | Create | Best DeepLabV3+ weights |
| `results/models/deeplab_results.csv` | Create | mIoU, pixel acc, per-class IoU |
| `results/models/deeplab_pred_grid.png` | Create | 4×4 grid of val predictions |
| `results/models/yolo_seg_best.pt` | Create | Best YOLOv8n-seg weights |
| `results/models/yolo_seg_results.csv` | Create | mask mAP50, box mAP50 |
| `results/models/yolo_seg_confusion.png` | Create | YOLO-seg confusion matrix |

**Dataset locations (not tracked in git):**
- `D:\Claude\datasets\ppe_seg\semantic\{train,val}\{images,masks}/` — DeepLabV3+ format
- `D:\Claude\datasets\ppe_seg\instance\{train,val}\{images,labels}/` + `ppe_seg_inst.yaml` — YOLOv8-seg format

---

### Task A: Install dependency and data preparation

**Files:**
- Create: `src/ppe_seg_data_prep.py`

- [ ] **Step A-1: Verify `datasets` package is installed**

```powershell
& "C:\Program Files\PyManager\python3.exe" -c "import datasets; print(datasets.__version__)"
```

Expected: version string (e.g. `2.x.x`). If ImportError:
```powershell
& "C:\Program Files\PyManager\python3.exe" -m pip install datasets
```

- [ ] **Step A-2: Copy sam2_b.pt into the worktree directory so scripts can find it**

The SAM2 checkpoint lives in the main worktree. Scripts run from the worktree directory, so copy or confirm it:

```powershell
if (-not (Test-Path "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794\sam2_b.pt")) {
    Copy-Item "D:\Claude\PPE-Detection\sam2_b.pt" `
              "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794\sam2_b.pt"
}
```

Expected: no error; file present in worktree root.

- [ ] **Step A-3: Create `src/ppe_seg_data_prep.py`**

```python
"""
PPE Segmentation Dataset Preparation
=====================================
Builds two pixel-level datasets from:
  1. Existing MinhNKB data (XML annotations + pre-generated SAM2 masks)
  2. keremberke/protective-equipment-detection (HuggingFace, ~2000 images,
     SAM2 box-prompted mask generation for helmet/no_helmet classes only)

Outputs:
  Semantic (DeepLabV3+):
    D:/Claude/datasets/ppe_seg/semantic/{train,val}/{images,masks}/
    Each mask is a single-channel uint8 PNG where pixel value = class index:
      0=background  1=full_ppe  2=helmet  3=no_ppe  4=partial_ppe  5=safety_vest

  Instance (YOLOv8-seg):
    D:/Claude/datasets/ppe_seg/instance/{train,val}/{images,labels}/
    D:/Claude/datasets/ppe_seg/instance/ppe_seg_inst.yaml
    Each .txt label file: one line per instance = class_id poly_x1 poly_y1 ...
    Coordinates are normalised 0-1.
"""

import os
import csv
import json
import random
import xml.etree.ElementTree as ET
import warnings
import numpy as np
import cv2
from pathlib import Path

warnings.filterwarnings('ignore')
random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
for _cand in ["D:/datasets", "D:/Claude/datasets",
              str(Path(__file__).resolve().parents[2] / "datasets")]:
    if os.path.exists(os.path.join(_cand, "jomarkow")):
        DATASETS = _cand
        break
else:
    raise FileNotFoundError("datasets/ not found")

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
MASK_DIR    = os.path.join(DATASETS, "ppe_masks")          # existing SAM2 binary masks
MINHNKB_IMG = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-data")
MINHNKB_ANN = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-annotations-new")

SEG_ROOT     = os.path.join(DATASETS, "ppe_seg")
SEM_ROOT     = os.path.join(SEG_ROOT, "semantic")
INST_ROOT    = os.path.join(SEG_ROOT, "instance")

SAM2_CKPT = os.path.join(PROJECT_DIR, "sam2_b.pt")

# ── Class definitions (alphabetical order matches existing project) ─────────
# Semantic mask pixel values:  0=bg  1=full_ppe  2=helmet  3=no_ppe  4=partial_ppe  5=safety_vest
CLASSES   = ["full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]
CLS2IDX   = {c: i + 1 for i, c in enumerate(CLASSES)}   # 1-indexed; 0=background

# Keremberke → our class (only helmet-related classes used; others skipped)
KEREMBERKE_MAP = {
    "helmet":    "helmet",
    "no_helmet": "no_ppe",
}

MINHNKB_MAP = {
    "helmet":                     "helmet",
    "safety vest":                "safety_vest",
    "person with full safety":    "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety":      "no_ppe",
}

VAL_FRAC   = 0.15   # 15% held out for validation
MAX_HF     = 2000   # max keremberke images to process (SAM2 is ~1.5s/image)


# ── Directory setup ────────────────────────────────────────────────────────
def _makedirs():
    for split in ("train", "val"):
        for ds_root in (SEM_ROOT, INST_ROOT):
            for sub in ("images", "masks" if ds_root == SEM_ROOT else "labels"):
                os.makedirs(os.path.join(ds_root, split, sub), exist_ok=True)

# ── SAM2 helper ────────────────────────────────────────────────────────────
_sam_model = None

def get_sam():
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        _sam_model = SAM(SAM2_CKPT)
        print(f"  SAM2 loaded from {SAM2_CKPT}")
    return _sam_model


def sam_mask_from_box(img_bgr, x1, y1, x2, y2):
    """Run SAM2 with a box prompt; return largest binary mask (H×W bool) or None."""
    try:
        sam = get_sam()
        results = sam(img_bgr, bboxes=[[x1, y1, x2, y2]], verbose=False)
        if results and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # (N, H, W) float
            areas = masks.sum(axis=(1, 2))
            best  = masks[areas.argmax()] > 0.5
            return best.astype(np.uint8)
    except Exception as e:
        print(f"    SAM2 error: {e}")
    return None


# ── Mask → polygon (YOLO normalised coords) ────────────────────────────────
def mask_to_polygon(binary_mask, img_w, img_h):
    """Return largest polygon as flat list of normalised (x, y) coords, or None."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 3:
        return None
    # Simplify to reduce point count
    epsilon = 0.005 * cv2.arcLength(c, True)
    approx  = cv2.approxPolyDP(c, epsilon, True)
    pts     = approx.reshape(-1, 2).astype(float)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    return pts.flatten().tolist()


# ── Write helpers ──────────────────────────────────────────────────────────
def write_sample(split, stem, img_bgr, semantic_mask, instances):
    """
    Write one sample to both semantic and instance dataset directories.

    Args:
        split:         'train' or 'val'
        stem:          unique filename stem (no extension)
        img_bgr:       BGR image array
        semantic_mask: H×W uint8 class-index mask (0=bg, 1-5=PPE class)
        instances:     list of (cls_idx, binary_mask_HW) tuples
    """
    img_path  = os.path.join(SEM_ROOT,  split, "images", stem + ".jpg")
    mask_path = os.path.join(SEM_ROOT,  split, "masks",  stem + ".png")
    inst_img  = os.path.join(INST_ROOT, split, "images", stem + ".jpg")
    inst_lbl  = os.path.join(INST_ROOT, split, "labels", stem + ".txt")

    cv2.imwrite(img_path,  img_bgr)
    cv2.imwrite(mask_path, semantic_mask)
    cv2.imwrite(inst_img,  img_bgr)

    h, w = img_bgr.shape[:2]
    lines = []
    for cls_idx, bmask in instances:
        poly = mask_to_polygon(bmask, w, h)
        if poly is None or len(poly) < 6:
            continue
        coords = " ".join(f"{v:.6f}" for v in poly)
        # YOLO-seg class is 0-indexed: cls_idx is 1-5, so subtract 1
        lines.append(f"{cls_idx - 1} {coords}")

    with open(inst_lbl, "w") as f:
        f.write("\n".join(lines))


# ── Source 1: MinhNKB (existing SAM2 binary masks + XML class labels) ──────
def process_minhnkb(split_map):
    """
    Build semantic + instance samples from MinhNKB.

    For each image that has a pre-generated SAM2 binary mask in MASK_DIR,
    intersect every bounding-box annotation with the foreground mask to get
    per-instance pixel boundaries, then assign the bounding-box class label.

    split_map: dict of img_stem → 'train'|'val'
    """
    xml_files = sorted(f for f in os.listdir(MINHNKB_ANN) if f.endswith(".xml"))
    processed = 0

    for xf in xml_files:
        stem = os.path.splitext(xf)[0]
        if stem not in split_map:
            continue
        split = split_map[stem]

        try:
            root = ET.parse(os.path.join(MINHNKB_ANN, xf)).getroot()
        except ET.ParseError:
            continue

        # Find source image
        fname = root.findtext("filename", "")
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            for cand in [os.path.join(MINHNKB_IMG, fname),
                         os.path.join(MINHNKB_IMG, stem + ext)]:
                if cand and os.path.exists(cand):
                    img_path = cand
                    break
            if img_path:
                break
        if not img_path:
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_h, img_w = img_bgr.shape[:2]

        # Load pre-generated SAM2 binary mask
        mask_file = os.path.join(MASK_DIR, stem + "_mask.png")
        scene_fg  = None
        if os.path.exists(mask_file):
            raw = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                scene_fg = (raw > 127).astype(np.uint8)

        # Build semantic mask and instance list from annotations
        semantic  = np.zeros((img_h, img_w), dtype=np.uint8)  # 0 = background
        instances = []

        for obj in root.findall("object"):
            raw_label = obj.findtext("name", "").strip().lower()
            cls_name  = MINHNKB_MAP.get(raw_label)
            if cls_name is None:
                continue
            cls_idx = CLS2IDX[cls_name]

            bb = obj.find("bndbox")
            if bb is None:
                continue
            x1 = max(0,     int(float(bb.findtext("xmin", "0"))))
            y1 = max(0,     int(float(bb.findtext("ymin", "0"))))
            x2 = min(img_w, int(float(bb.findtext("xmax", str(img_w)))))
            y2 = min(img_h, int(float(bb.findtext("ymax", str(img_h)))))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue

            # Instance mask: SAM2 foreground within bbox, or bbox fill fallback
            inst_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            if scene_fg is not None:
                roi          = scene_fg[y1:y2, x1:x2]
                inst_roi     = inst_mask[y1:y2, x1:x2]
                inst_roi[:] = roi
            else:
                inst_mask[y1:y2, x1:x2] = 1   # bbox fill fallback

            # Semantic: overwrite with this class (last writer wins for overlaps)
            semantic[inst_mask == 1] = cls_idx
            instances.append((cls_idx, inst_mask))

        if not instances:
            continue

        write_sample(split, f"mnk_{stem}", img_bgr, semantic, instances)
        processed += 1

    print(f"  MinhNKB: {processed} images written")


# ── Source 2: keremberke HuggingFace dataset ───────────────────────────────
def process_keremberke(split_map_hf):
    """
    Download keremberke/protective-equipment-detection, sample MAX_HF images,
    generate SAM2 masks for helmet/no_helmet boxes, write samples.

    split_map_hf: list of (example, split) tuples already sampled
    """
    processed = 0

    for example, split in split_map_hf:
        img_pil = example["image"]
        if img_pil is None:
            continue
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]

        objects = example.get("objects", {})
        bboxes_raw = objects.get("bbox", [])
        cats       = objects.get("category", [])

        # Map category ids → names using the dataset's id2label
        # keremberke stores categories as integer ids in 'category' field
        # We resolve with the dataset features
        semantic  = np.zeros((img_h, img_w), dtype=np.uint8)
        instances = []

        for bbox, cat_name in zip(bboxes_raw, cats):
            # cat_name comes in as a string class name in the loaded HF dataset
            mapped = KEREMBERKE_MAP.get(str(cat_name).lower())
            if mapped is None:
                continue
            cls_idx = CLS2IDX[mapped]

            # COCO bbox: [x, y, w, h]
            x1 = max(0,     int(bbox[0]))
            y1 = max(0,     int(bbox[1]))
            x2 = min(img_w, int(bbox[0] + bbox[2]))
            y2 = min(img_h, int(bbox[1] + bbox[3]))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue

            inst_mask = sam_mask_from_box(img_bgr, x1, y1, x2, y2)
            if inst_mask is None:
                # Fallback: bbox fill
                inst_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                inst_mask[y1:y2, x1:x2] = 1

            semantic[inst_mask == 1] = cls_idx
            instances.append((cls_idx, inst_mask))

        if not instances:
            continue

        stem = f"ke_{processed:05d}"
        write_sample(split, stem, img_bgr, semantic, instances)
        processed += 1
        if processed % 100 == 0:
            print(f"  keremberke: {processed}/{len(split_map_hf)} done …")

    print(f"  keremberke: {processed} images written")


# ── YOLO YAML ─────────────────────────────────────────────────────────────
def write_yolo_yaml():
    yaml_path = os.path.join(INST_ROOT, "ppe_seg_inst.yaml")
    content = f"""path: {INST_ROOT.replace(chr(92), '/')}
train: train/images
val:   val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  YOLO YAML: {yaml_path}")
    return yaml_path


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("PPE SEGMENTATION DATA PREP")
    print("=" * 65)

    _makedirs()

    # ── MinhNKB split map ──────────────────────────────────────────────────
    xml_stems = [os.path.splitext(f)[0]
                 for f in os.listdir(MINHNKB_ANN) if f.endswith(".xml")]
    random.shuffle(xml_stems)
    n_val = max(1, int(len(xml_stems) * VAL_FRAC))
    split_map = {s: ("val" if i < n_val else "train")
                 for i, s in enumerate(xml_stems)}

    print(f"\n[1/3] MinhNKB ({len(xml_stems)} images, val={n_val}) …")
    process_minhnkb(split_map)

    # ── keremberke HF download + SAM2 ─────────────────────────────────────
    print(f"\n[2/3] keremberke HF dataset (sampling {MAX_HF} images) …")
    try:
        from datasets import load_dataset
        hf_ds = load_dataset("keremberke/protective-equipment-detection",
                              name="full", trust_remote_code=True)
        train_pool = list(hf_ds["train"])
        val_pool   = list(hf_ds["validation"])
        random.shuffle(train_pool); random.shuffle(val_pool)

        n_hf_val   = max(1, int(MAX_HF * VAL_FRAC))
        n_hf_train = MAX_HF - n_hf_val

        split_map_hf = (
            [(ex, "train") for ex in train_pool[:n_hf_train]] +
            [(ex, "val")   for ex in val_pool[:n_hf_val]]
        )
        process_keremberke(split_map_hf)
    except Exception as e:
        print(f"  WARNING: keremberke download failed ({e}) — using MinhNKB only")

    # ── YOLO YAML ─────────────────────────────────────────────────────────
    print("\n[3/3] Writing YOLO YAML …")
    yaml_path = write_yolo_yaml()

    # ── Summary ───────────────────────────────────────────────────────────
    for split in ("train", "val"):
        n = len(os.listdir(os.path.join(SEM_ROOT, split, "images")))
        print(f"  Semantic {split}: {n} images")
    print(f"\nDataset ready. YOLO config: {yaml_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step A-4: Run data preparation**

```powershell
cd "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794"
& "C:\Program Files\PyManager\python3.exe" src/ppe_seg_data_prep.py 2>&1
```

Expected output (approximate):
```
PPE SEGMENTATION DATA PREP
[1/3] MinhNKB (1613 images, val=242) …
  MinhNKB: ~1370 images written
[2/3] keremberke HF dataset (sampling 2000 images) …
  keremberke: 100/1700 done …
  ...
  keremberke: ~1500 images written
[3/3] Writing YOLO YAML …
  Semantic train: ~2500 images
  Semantic val:   ~400 images
```

SAM2 mask generation for ~1700 keremberke train images takes ~40–50 minutes on RTX 5070.

- [ ] **Step A-5: Commit data prep script**

```bash
cd "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794"
git add src/ppe_seg_data_prep.py
git commit -m "feat: add segmentation dataset prep (MinhNKB + keremberke + SAM2 masks)"
```

---

### Task B: DeepLabV3+ Semantic Segmentation

**Files:**
- Create: `src/ppe_deeplab_train.py`

- [ ] **Step B-1: Create `src/ppe_deeplab_train.py`**

```python
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
    print(f"Pixel Acc: {pixel_acc:.4f}")

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

    # ── Prediction grid (4×4 sample) ──
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

    val_items = [val_ds[i] for i in range(min(8, len(val_ds)))]
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    model.eval()
    with torch.no_grad():
        for i in range(min(8, len(val_items))):
            img_t, mask_t = val_items[i]
            pred = model(img_t.unsqueeze(0).to(DEVICE))["out"].argmax(1)[0].cpu()
            img_np = ((img_t * std_t + mean_t).permute(1,2,0).numpy() * 255).clip(0,255).astype(np.uint8)
            row, base_col = divmod(i, 2)
            axes[row*2][base_col*2].imshow(img_np); axes[row*2][base_col*2].axis("off")
            axes[row*2][base_col*2].set_title("Image", fontsize=8)
            pred_rgb = COLORS[pred.numpy()]
            axes[row*2][base_col*2+1].imshow(pred_rgb); axes[row*2][base_col*2+1].axis("off")
            axes[row*2][base_col*2+1].set_title("Pred", fontsize=8)

    plt.suptitle("DeepLabV3+ Predictions", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "deeplab_pred_grid.png"), dpi=150)
    plt.close()
    print(f"Saved plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step B-2: Run DeepLabV3+ training**

```powershell
cd "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794"
& "C:\Program Files\PyManager\python3.exe" src/ppe_deeplab_train.py 2>&1
```

Expected: 30 epochs, ~3–5 min each on RTX 5070. Final output:
```
Ep  30/30  train_loss=~0.35  val_mIoU=~0.45  best=~0.48
Per-class IoU: helmet ~0.65, no_ppe ~0.55, safety_vest ~0.45 ...
Mean IoU: ~0.45–0.55
```

- [ ] **Step B-3: Commit DeepLabV3+ results**

```bash
git add src/ppe_deeplab_train.py results/models/deeplab_results.csv \
        results/models/deeplab_model.pth results/models/deeplab_training.png \
        results/models/deeplab_confusion.png results/models/deeplab_pred_grid.png
git commit -m "feat: DeepLabV3+ semantic PPE segmentation (6-class pixel labelling)"
```

Then push to main:
```bash
cd "D:\Claude\PPE-Detection" && git merge --ff-only claude/flamboyant-payne-c40794 && git push origin main && git checkout -
```

---

### Task C: YOLOv8n-seg Instance Segmentation

**Files:**
- No Python script needed — uses `yolo train` CLI
- Create: `results/models/yolo_seg_results.csv` (written manually from training output)

- [ ] **Step C-1: Verify YOLO-seg dataset**

```powershell
$yaml = "D:\Claude\datasets\ppe_seg\instance\ppe_seg_inst.yaml"
Get-Content $yaml
$n_train = (Get-ChildItem "D:\Claude\datasets\ppe_seg\instance\train\images").Count
$n_val   = (Get-ChildItem "D:\Claude\datasets\ppe_seg\instance\val\images").Count
Write-Host "Train: $n_train  Val: $n_val"
```

Expected: YAML shows 5 classes; train > 1000 images.

- [ ] **Step C-2: Train YOLOv8n-seg**

```powershell
cd "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794"
yolo train model=yolov8n-seg.pt `
     data="D:/Claude/datasets/ppe_seg/instance/ppe_seg_inst.yaml" `
     epochs=30 imgsz=640 batch=16 device=0 `
     name=yolov8_ppe_seg_prod project=runs/segment 2>&1 | Select-Object -Last 30
```

Expected: 30 epochs, ~15–25 min on RTX 5070.
Final line contains `mask mAP50`, `box mAP50`, `Precision`, `Recall`.

- [ ] **Step C-3: Copy results to results/models/**

```powershell
$src = Get-ChildItem "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794\runs\segment" -Recurse -Name "confusion_matrix.png" | Select-Object -First 1
$run = Split-Path (Join-Path "runs\segment" $src) -Parent
$dst = "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794\results\models"

Copy-Item "$run\confusion_matrix.png"      "$dst\yolo_seg_confusion.png"      -Force
Copy-Item "$run\BoxPR_curve.png"           "$dst\yolo_seg_pr_curve.png"       -Force
Copy-Item "$run\results.png"               "$dst\yolo_seg_results_plot.png"   -Force
Copy-Item "$run\weights\best.pt"           "$dst\yolo_seg_best.pt"            -Force
```

- [ ] **Step C-4: Write yolo_seg_results.csv from training output**

Read the printed metrics from Step C-2 output and create the CSV:

```python
# Run this snippet to write the CSV — fill in the actual values from training output
import csv, os
out = "D:/Claude/PPE-Detection/.claude/worktrees/flamboyant-payne-c40794/results/models/yolo_seg_results.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model","Task","mask_mAP50","box_mAP50","Precision","Recall","Train_Time(s)","Architecture","Params_K","Notes"])
    # Replace values below with actual output from Step C-2
    w.writerow(["YOLOv8n-seg (PPE)","instance_seg",
                "FILL_MASK_MAP50","FILL_BOX_MAP50","FILL_PREC","FILL_REC","FILL_TIME",
                "YOLO","3200","5-class instance segmentation"])
print(f"Written to {out}")
```

- [ ] **Step C-5: Commit YOLOv8n-seg results**

```bash
git add results/models/yolo_seg_best.pt results/models/yolo_seg_results.csv \
        results/models/yolo_seg_confusion.png results/models/yolo_seg_pr_curve.png \
        results/models/yolo_seg_results_plot.png
git commit -m "feat: YOLOv8n-seg instance PPE segmentation training"
```

Push to main:
```bash
cd "D:\Claude\PPE-Detection" && git merge --ff-only claude/flamboyant-payne-c40794 && git push origin main && git checkout -
```

---

### Task D: Update Experiment Comparison + Writeup + Final Push

**Files:**
- Modify: `src/ppe_experiment_comparison.py`
- Modify: `reports/generate_assignment_report.py`
- Overwrite: `docs/Final_Project_Writeup.docx`

- [ ] **Step D-1: Add segmentation loaders to experiment comparison**

In `src/ppe_experiment_comparison.py`, after the `yolo_e2e_results.csv` block add:

```python
    # ------------------------------------------------------------------
    # 8. deeplab_results.csv  (DeepLabV3+ semantic segmentation)
    # ------------------------------------------------------------------
    path = os.path.join(out_dir, "deeplab_results.csv")
    try:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                'Model':        str(r['Model']),
                'Task':         'semantic_seg',
                'Accuracy':     np.nan,
                'mIoU':         float(r['mIoU']) if pd.notna(r.get('mIoU')) else np.nan,
                'Macro_F1':     np.nan,
                'Weighted_F1':  np.nan,
                'Architecture': 'DeepLab',
                'Params_K':     39000,
                'Train_Time_s': float(r['Train_Time(s)']) if pd.notna(r.get('Train_Time(s)')) else np.nan,
                'Notes':        '6-class semantic segmentation',
            })
        print(f"  Loaded {len(df)} rows from deeplab_results.csv")
    except Exception as exc:
        print(f"  WARNING: could not load deeplab_results.csv -- {exc}")

    # ------------------------------------------------------------------
    # 9. yolo_seg_results.csv  (YOLOv8n-seg instance segmentation)
    # ------------------------------------------------------------------
    path = os.path.join(out_dir, "yolo_seg_results.csv")
    try:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                'Model':        str(r['Model']),
                'Task':         'instance_seg',
                'Accuracy':     float(r['mask_mAP50']) if pd.notna(r.get('mask_mAP50')) else np.nan,
                'mIoU':         np.nan,
                'Macro_F1':     np.nan,
                'Weighted_F1':  np.nan,
                'Architecture': 'YOLO',
                'Params_K':     3200,
                'Train_Time_s': float(r['Train_Time(s)']) if pd.notna(r.get('Train_Time(s)')) else np.nan,
                'Notes':        str(r.get('Notes', '')),
            })
        print(f"  Loaded {len(df)} rows from yolo_seg_results.csv")
    except Exception as exc:
        print(f"  WARNING: could not load yolo_seg_results.csv -- {exc}")
```

Also add `'DeepLab': '#17BECF'` to `ARCH_COLOURS`, `'DeepLab': 39000` to `PARAM_COUNTS`, and in `plot_accuracy_comparison` add `'instance_seg'` to the detection block (uses mask_mAP50 stored in Accuracy).

- [ ] **Step D-2: Run experiment comparison**

```powershell
cd "D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794"
& "C:\Program Files\PyManager\python3.exe" src/ppe_experiment_comparison.py 2>&1
```

Expected: 19 model results loaded (17 previous + DeepLab + YOLO-seg).

- [ ] **Step D-3: Update and regenerate writeup**

In `reports/generate_assignment_report.py`, add two rows to the results table and a pixel-wise analysis section, then run:

```powershell
& "C:\Program Files\PyManager\python3.exe" reports/generate_assignment_report.py 2>&1
```

Expected: `Writeup saved to docs/Final_Project_Writeup.docx`

- [ ] **Step D-4: Final commit and push**

```bash
git add src/ppe_experiment_comparison.py reports/generate_assignment_report.py \
        docs/Final_Project_Writeup.docx \
        results/models/experiment_comparison.png \
        results/models/experiment_comparison_full.csv \
        results/models/experiment_table.tex \
        results/models/experiment_f1_heatmap.png
git commit -m "docs: add pixel-wise segmentation results to experiment comparison and writeup"
cd "D:\Claude\PPE-Detection" && git merge --ff-only claude/flamboyant-payne-c40794 && git push origin main
```
