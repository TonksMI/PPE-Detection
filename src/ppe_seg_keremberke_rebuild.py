"""
Keremberke Full Dataset Builder
=================================
Rebuilds the ppe_seg dataset from scratch using all three keremberke zip files
and the native 10-class keremberke schema. Drops the MinhNKB 5-class schema entirely.

Keremberke classes used:
  0 = background
  1 = helmet
  2 = no_helmet
  3 = glove
  4 = no_glove
  5 = goggles
  6 = no_goggles
  7 = mask
  8 = no_mask
  9 = shoes
  10 = no_shoes

Zip usage:
  train.zip   → mostly glove/no_glove/goggles (6473 images)
  test.zip    → mostly helmet/no_helmet (1935 images)
  valid.zip   → mixed all classes (3570 images)

All images with at least one relevant annotation are pooled, capped at MAX_IMAGES,
shuffled (seed=42), split 85/15 train/val, then SAM2 masks are generated.

Outputs (wipe and rebuild):
  D:/Claude/datasets/ppe_seg_ke/semantic/{train,val}/{images,masks}/
  D:/Claude/datasets/ppe_seg_ke/instance/{train,val}/{images,labels}/
  D:/Claude/datasets/ppe_seg_ke/instance/ppe_seg_ke.yaml
"""

import io
import os
import json
import random
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")
random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
for _cand in ["D:/datasets", "D:/Claude/datasets",
              str(Path(__file__).resolve().parents[2] / "datasets")]:
    if os.path.exists(os.path.join(_cand, "jomarkow")):
        DATASETS = _cand
        break
else:
    raise FileNotFoundError("datasets/ not found")

PROJECT_DIR  = str(Path(__file__).resolve().parent.parent)
WORKTREE_DIR = PROJECT_DIR

SEG_ROOT  = os.path.join(DATASETS, "ppe_seg_ke")
SEM_ROOT  = os.path.join(SEG_ROOT, "semantic")
INST_ROOT = os.path.join(SEG_ROOT, "instance")

ZIP_FILES = [
    os.path.join(WORKTREE_DIR, "train.zip"),
    os.path.join(WORKTREE_DIR, "test.zip"),
    os.path.join(WORKTREE_DIR, "valid.zip"),
]

SAM2_CKPT = os.path.join(PROJECT_DIR, "sam2_b.pt")

# ── Class schema ───────────────────────────────────────────────────────────
# All 10 keremberke PPE classes (background = 0)
CLASSES = [
    "helmet", "no_helmet",
    "glove",  "no_glove",
    "goggles","no_goggles",
    "mask",   "no_mask",
    "shoes",  "no_shoes",
]
CLS2IDX = {c: i + 1 for i, c in enumerate(CLASSES)}  # 1-indexed; 0=background
N_CLASSES = len(CLASSES) + 1  # 11 total (background + 10)

# All category names in keremberke map directly to the same name
KE_MAP = {c: c for c in CLASSES}

MAX_IMAGES = 4000   # cap total — SAM2 at ~1.5s/image ≈ 100 minutes
VAL_FRAC   = 0.15


# ── SAM2 ──────────────────────────────────────────────────────────────────
_predictor = None

def _get_predictor():
    global _predictor
    if _predictor is None:
        from ultralytics import SAM
        _predictor = SAM(SAM2_CKPT)
    return _predictor

def sam_mask_from_box(img_bgr, x1, y1, x2, y2):
    try:
        pred = _get_predictor()
        results = pred(img_bgr, bboxes=[[x1, y1, x2, y2]], verbose=False)
        if results and results[0].masks is not None:
            m = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            h, w = img_bgr.shape[:2]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            return m
    except Exception:
        pass
    return None


# ── Directory setup ────────────────────────────────────────────────────────
def _makedirs():
    for split in ("train", "val"):
        os.makedirs(os.path.join(SEM_ROOT,  split, "images"), exist_ok=True)
        os.makedirs(os.path.join(SEM_ROOT,  split, "masks"),  exist_ok=True)
        os.makedirs(os.path.join(INST_ROOT, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(INST_ROOT, split, "labels"), exist_ok=True)


def write_sample(split, stem, img_bgr, semantic, instances):
    img_h, img_w = img_bgr.shape[:2]

    # Semantic
    cv2.imwrite(os.path.join(SEM_ROOT, split, "images", f"{stem}.jpg"), img_bgr)
    cv2.imwrite(os.path.join(SEM_ROOT, split, "masks",  f"{stem}.png"), semantic)

    # Instance
    cv2.imwrite(os.path.join(INST_ROOT, split, "images", f"{stem}.jpg"), img_bgr)
    lines = []
    for cls_idx, mask in instances:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 3:
            continue
        pts = cnt.squeeze()
        if pts.ndim < 2 or len(pts) < 3:
            continue
        coords = " ".join(f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in pts)
        lines.append(f"{cls_idx - 1} {coords}")   # YOLO 0-indexed

    with open(os.path.join(INST_ROOT, split, "labels", f"{stem}.txt"), "w") as f:
        f.write("\n".join(lines))


def write_yolo_yaml():
    yaml_path = os.path.join(INST_ROOT, "ppe_seg_ke.yaml")
    content = f"""path: {INST_ROOT.replace(chr(92), '/')}
train: train/images
val:   val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  YAML: {yaml_path}")
    return yaml_path


# ── Collect entries from all zips ──────────────────────────────────────────
def collect_entries():
    """
    Returns list of (zip_path, image_filename, annotations) tuples.
    annotations = list of (bbox_xywh, class_name).
    Only images with >= 1 annotation in CLASSES are included.
    """
    entries = []
    for zip_path in ZIP_FILES:
        if not os.path.exists(zip_path):
            print(f"  WARNING: {zip_path} not found — skipping")
            continue

        with zipfile.ZipFile(zip_path) as zf:
            data = json.loads(zf.read("_annotations.coco.json"))

        cats    = {c["id"]: c["name"] for c in data["categories"]}
        rel_ids = {i for i, n in cats.items() if n in KE_MAP}
        img_map = {img["id"]: img["file_name"] for img in data["images"]}

        img_anns = defaultdict(list)
        for ann in data["annotations"]:
            if ann["category_id"] in rel_ids:
                img_anns[ann["image_id"]].append(
                    (ann["bbox"], cats[ann["category_id"]])
                )

        z_name  = os.path.basename(zip_path)
        n_imgs  = len(img_anns)
        entries += [(zip_path, img_map[iid], anns) for iid, anns in img_anns.items()]
        print(f"  {z_name}: {n_imgs} usable images")

    return entries


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("KEREMBERKE FULL DATASET BUILDER (10 native classes)")
    print("=" * 65)

    # Wipe and rebuild
    if os.path.exists(SEG_ROOT):
        print(f"\nClearing existing dataset at {SEG_ROOT} ...")
        shutil.rmtree(SEG_ROOT)
    _makedirs()

    entries = collect_entries()
    print(f"\nTotal usable images across all zips: {len(entries)}")

    random.shuffle(entries)
    entries = entries[:MAX_IMAGES]
    print(f"Capped to {len(entries)} images (MAX_IMAGES={MAX_IMAGES})")

    n_val = max(1, int(len(entries) * VAL_FRAC))
    print(f"Split: {len(entries) - n_val} train / {n_val} val\n")

    zip_cache  = {}
    processed  = 0
    skipped    = 0

    for idx, (zip_path, fname, anns) in enumerate(entries):
        split = "val" if idx < n_val else "train"
        stem  = f"ke_{idx:05d}"

        if zip_path not in zip_cache:
            zip_cache[zip_path] = zipfile.ZipFile(zip_path)
        zf = zip_cache[zip_path]

        try:
            img_bytes = zf.read(fname)
        except KeyError:
            skipped += 1
            continue

        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            skipped += 1
            continue

        img_h, img_w = img_bgr.shape[:2]
        semantic  = np.zeros((img_h, img_w), dtype=np.uint8)
        instances = []

        for bbox, cat_name in anns:
            mapped = KE_MAP.get(cat_name)
            if mapped is None:
                continue
            cls_idx = CLS2IDX[mapped]

            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(img_w, int(bbox[0] + bbox[2]))
            y2 = min(img_h, int(bbox[1] + bbox[3]))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue

            inst_mask = sam_mask_from_box(img_bgr, x1, y1, x2, y2)
            if inst_mask is None:
                inst_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                inst_mask[y1:y2, x1:x2] = 1

            semantic[inst_mask == 1] = cls_idx
            instances.append((cls_idx, inst_mask))

        if not instances:
            skipped += 1
            continue

        write_sample(split, stem, img_bgr, semantic, instances)
        processed += 1

        if processed % 100 == 0:
            print(f"  [{processed}/{len(entries)}] processed ...")

    for zf in zip_cache.values():
        zf.close()

    write_yolo_yaml()

    print(f"\nDone.")
    print(f"  Written:  {processed} images ({skipped} skipped)")
    # Final count
    for split in ("train", "val"):
        n = len(os.listdir(os.path.join(SEM_ROOT, split, "images")))
        print(f"  {split}: {n} images")


if __name__ == "__main__":
    main()
