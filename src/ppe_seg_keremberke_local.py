"""
Keremberke Local Zip Processor
================================
Reads keremberke/protective-equipment-detection COCO zip files (test.zip, valid.zip)
directly from the project worktree and appends helmet/no_helmet samples to the
existing ppe_seg semantic + instance datasets.

Only test.zip and valid.zip have helmet/no_helmet annotations (train.zip is all
gloves/goggles).  All images with at least one helmet or no_helmet box are pooled,
shuffled (seed=42), and split 85/15 train/val then added to the existing dataset.

Run AFTER ppe_seg_data_prep.py has already built the MinhNKB baseline.
"""

import io
import os
import json
import random
import zipfile
from pathlib import Path

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

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)

SEG_ROOT  = os.path.join(DATASETS, "ppe_seg")
SEM_ROOT  = os.path.join(SEG_ROOT, "semantic")
INST_ROOT = os.path.join(SEG_ROOT, "instance")

# Zip files live in the worktree root (alongside CLAUDE.md)
WORKTREE_DIR = PROJECT_DIR
ZIP_FILES = [
    os.path.join(WORKTREE_DIR, "test.zip"),
    os.path.join(WORKTREE_DIR, "valid.zip"),
    # train.zip has no helmet/no_helmet annotations — skip
]

SAM2_CKPT = os.path.join(PROJECT_DIR, "sam2_b.pt")

# ── Class map (matches ppe_seg_data_prep.py) ───────────────────────────────
CLASSES  = ["full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]
CLS2IDX  = {c: i + 1 for i, c in enumerate(CLASSES)}   # 1-indexed; 0=background

KEREMBERKE_MAP = {
    "helmet":    "helmet",
    "no_helmet": "no_ppe",
}

VAL_FRAC = 0.15
MAX_IMAGES = 2000  # cap (SAM2 ~1.5 s/image)


# ── SAM2 helper ────────────────────────────────────────────────────────────
_sam2_predictor = None

def _get_predictor():
    global _sam2_predictor
    if _sam2_predictor is None:
        from ultralytics import SAM
        _sam2_predictor = SAM(SAM2_CKPT)
    return _sam2_predictor

def sam_mask_from_box(img_bgr, x1, y1, x2, y2):
    """Return HxW uint8 binary mask from SAM2 box prompt, or None on failure."""
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


# ── Write helpers (mirrors ppe_seg_data_prep.py) ───────────────────────────
def _sem_dir(split):
    return os.path.join(SEM_ROOT, split)

def _inst_dir(split):
    return os.path.join(INST_ROOT, split)

def write_sample(split, stem, img_bgr, semantic, instances):
    """Write one semantic mask + instance label set (same format as data_prep)."""
    # Semantic
    img_path  = os.path.join(_sem_dir(split), "images", f"{stem}.jpg")
    mask_path = os.path.join(_sem_dir(split), "masks",  f"{stem}.png")
    cv2.imwrite(img_path, img_bgr)
    cv2.imwrite(mask_path, semantic)

    # Instance
    inst_img  = os.path.join(_inst_dir(split), "images", f"{stem}.jpg")
    inst_lbl  = os.path.join(_inst_dir(split), "labels", f"{stem}.txt")
    cv2.imwrite(inst_img, img_bgr)

    h, w = img_bgr.shape[:2]
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
        coords = " ".join(
            f"{x / w:.6f} {y / h:.6f}" for x, y in pts
        )
        lines.append(f"{cls_idx - 1} {coords}")  # YOLO class id is 0-indexed

    with open(inst_lbl, "w") as f:
        f.write("\n".join(lines))


# ── Collect all relevant images from zips ─────────────────────────────────
def collect_entries():
    """
    Returns list of (zip_path, image_filename, annotations) tuples.
    annotations = list of (bbox_xywh, class_name).
    Only entries with at least one helmet/no_helmet annotation are included.
    """
    entries = []
    for zip_path in ZIP_FILES:
        if not os.path.exists(zip_path):
            print(f"  WARNING: {zip_path} not found — skipping")
            continue
        with zipfile.ZipFile(zip_path) as zf:
            data = json.loads(zf.read("_annotations.coco.json"))
        cats   = {c["id"]: c["name"] for c in data["categories"]}
        rel_ids = {i for i, n in cats.items() if n in KEREMBERKE_MAP}
        img_map = {img["id"]: img["file_name"] for img in data["images"]}

        # Group annotations by image
        from collections import defaultdict
        img_anns = defaultdict(list)
        for ann in data["annotations"]:
            if ann["category_id"] in rel_ids:
                img_anns[ann["image_id"]].append(
                    (ann["bbox"], cats[ann["category_id"]])
                )

        for img_id, anns in img_anns.items():
            entries.append((zip_path, img_map[img_id], anns))
        print(f"  {os.path.basename(zip_path)}: {len(img_anns)} usable images")

    return entries


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("KEREMBERKE LOCAL ZIP PROCESSOR")
    print("=" * 65)

    entries = collect_entries()
    print(f"\nTotal usable images: {len(entries)}")

    random.shuffle(entries)
    entries = entries[:MAX_IMAGES]

    n_val   = max(1, int(len(entries) * VAL_FRAC))
    val_set = set(range(n_val))

    # Check for existing stems to avoid overwriting
    existing = set()
    for split in ("train", "val"):
        img_dir = os.path.join(_sem_dir(split), "images")
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                existing.add(os.path.splitext(f)[0])

    processed = 0
    skipped   = 0

    zip_cache = {}  # keep open zip handles during processing

    for idx, (zip_path, fname, anns) in enumerate(entries):
        split = "val" if idx < n_val else "train"
        stem  = f"ke_{idx:05d}"

        if stem in existing:
            skipped += 1
            continue

        # Load image from zip
        if zip_path not in zip_cache:
            zip_cache[zip_path] = zipfile.ZipFile(zip_path)
        zf = zip_cache[zip_path]

        try:
            img_bytes = zf.read(fname)
        except KeyError:
            continue

        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        img_h, img_w = img_bgr.shape[:2]
        semantic  = np.zeros((img_h, img_w), dtype=np.uint8)
        instances = []

        for bbox, cat_name in anns:
            mapped = KEREMBERKE_MAP.get(cat_name)
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
            continue

        write_sample(split, stem, img_bgr, semantic, instances)
        processed += 1

        if processed % 50 == 0:
            print(f"  [{processed}/{len(entries)}] processed ...")

    for zf in zip_cache.values():
        zf.close()

    print(f"\nDone. Added {processed} new samples ({skipped} already existed).")
    print("Run ppe_deeplab_train.py and ppe_yolo_seg_train.py to retrain.")


if __name__ == "__main__":
    main()
