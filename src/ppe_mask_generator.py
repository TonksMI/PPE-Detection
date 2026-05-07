"""
PPE MASK GENERATOR
==================
One-time preprocessing script: uses Ultralytics SAM2 to auto-generate
pseudo-segmentation masks from bounding box annotations.

Outputs:
  - D:\\datasets\\ppe_masks\\{stem}_mask.png  (uint8, pixel = class_idx or 255)
  - results/models/mask_index.csv
  - results/models/mask_verification.png

These masks are used as ground-truth by ppe_unet_train.py (Task 4).

DO NOT run this script repeatedly — masks are written once and cached on disk.
"""

import os
import sys
import glob
import csv
import random
import warnings
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

warnings.filterwarnings('ignore')

# ── Path resolution (mirrors all other scripts in this project) ─────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)   # PPE-Detection/
BASE        = os.path.dirname(PROJECT_DIR)  # D:\Claude  or  /sessions/...

if os.path.exists("D:/datasets/jomarkow"):
    DATASETS = "D:/datasets"
    MASK_DIR = "D:/datasets/ppe_masks"
else:
    DATASETS = os.path.join(BASE, "datasets")
    MASK_DIR = os.path.join(BASE, "datasets", "ppe_masks")

MINHNKB_IMG = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-data")
MINHNKB_ANN = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-annotations-new")
JOMARK_IMG  = os.path.join(DATASETS, "jomarkow/images")
JOMARK_LBL  = os.path.join(DATASETS, "jomarkow/labels")

OUT_DIR = os.path.join(PROJECT_DIR, "results/models")
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

# ── Class definitions ────────────────────────────────────────────────────────
MINHNKB_MAP = {
    "helmet":                     "helmet",
    "safety vest":                "safety_vest",
    "person with full safety":    "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety":      "no_ppe",
}
JOMARKOW_MAP = {0: "helmet", 1: "no_ppe"}

ALL_CLASSES   = ["full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]
CLASS_TO_IDX  = {c: i for i, c in enumerate(ALL_CLASSES)}
# full_ppe=0, helmet=1, no_ppe=2, partial_ppe=3, safety_vest=4

MASK_COLORS = {
    0:   (128,   0, 128),  # full_ppe    → purple
    1:   ( 39, 174,  96),  # helmet      → green
    2:   (231,  76,  60),  # no_ppe      → red
    3:   (230, 126,  34),  # partial_ppe → orange
    4:   ( 41, 128, 185),  # safety_vest → blue
    255: (  0,   0,   0),  # background  → black
}

# ── Annotation parsing ───────────────────────────────────────────────────────

def _find_image(img_dir, stem, preferred=None):
    """Return the first existing image path for a given stem, or None."""
    if preferred and os.path.exists(preferred):
        return preferred
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def parse_minhnkb_annotations():
    """
    Parse Pascal VOC XML annotations from the MinhNKB dataset.

    Returns a list of dicts:
        {'img_path', 'cls', 'cls_idx', 'x1', 'y1', 'x2', 'y2', 'img_w', 'img_h'}
    """
    records = []
    xml_files = sorted(glob.glob(os.path.join(MINHNKB_ANN, "*.xml")))
    for xf in xml_files:
        try:
            root  = ET.parse(xf).getroot()
            fname = root.findtext("filename")
            stem  = os.path.splitext(os.path.basename(xf))[0]
            preferred = os.path.join(MINHNKB_IMG, fname) if fname else None
            ip = _find_image(MINHNKB_IMG, stem, preferred)
            if ip is None:
                continue

            sz_node = root.find("size")
            if sz_node is not None:
                img_w = int(sz_node.findtext("width",  "0") or 0)
                img_h = int(sz_node.findtext("height", "0") or 0)
            else:
                img_w = img_h = 0

            # Fallback: read actual dimensions from image header
            if img_w == 0 or img_h == 0:
                tmp = cv2.imread(ip)
                if tmp is None:
                    continue
                img_h, img_w = tmp.shape[:2]

            for obj in root.findall("object"):
                raw = obj.findtext("name", "").strip().lower()
                if raw not in MINHNKB_MAP:
                    continue
                cls = MINHNKB_MAP[raw]
                bb  = obj.find("bndbox")
                x1  = max(0,     int(float(bb.findtext("xmin", "0"))))
                y1  = max(0,     int(float(bb.findtext("ymin", "0"))))
                x2  = min(img_w, int(float(bb.findtext("xmax", str(img_w)))))
                y2  = min(img_h, int(float(bb.findtext("ymax", str(img_h)))))
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    continue
                records.append({
                    "img_path": ip,
                    "cls":      cls,
                    "cls_idx":  CLASS_TO_IDX[cls],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "img_w": img_w, "img_h": img_h,
                })
        except Exception:
            pass
    return records


def parse_jomarkow_annotations():
    """
    Parse YOLO-format TXT annotations from the Jomarkow dataset.
    Converts normalised (cx, cy, bw, bh) → pixel (x1, y1, x2, y2).

    Returns a list of dicts matching the MinhNKB schema.
    """
    records = []
    lbl_files = sorted(glob.glob(os.path.join(JOMARK_LBL, "*.txt")))
    for lf in lbl_files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        ip = _find_image(JOMARK_IMG, stem)
        if ip is None:
            continue
        try:
            img = cv2.imread(ip)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            with open(lf) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    if cid not in JOMARKOW_MAP:
                        continue
                    cls = JOMARKOW_MAP[cid]
                    cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                                      float(parts[3]), float(parts[4]))
                    x1 = max(0,     int((cx - bw / 2) * img_w))
                    y1 = max(0,     int((cy - bh / 2) * img_h))
                    x2 = min(img_w, int((cx + bw / 2) * img_w))
                    y2 = min(img_h, int((cy + bh / 2) * img_h))
                    if x2 <= x1 + 1 or y2 <= y1 + 1:
                        continue
                    records.append({
                        "img_path": ip,
                        "cls":      cls,
                        "cls_idx":  CLASS_TO_IDX[cls],
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "img_w": img_w, "img_h": img_h,
                    })
        except Exception:
            pass
    return records


def group_by_image(records):
    """
    Return an ordered dict: img_path → list of annotation dicts.
    Preserves the original record order within each image.
    """
    from collections import OrderedDict
    grouped = OrderedDict()
    for r in records:
        grouped.setdefault(r["img_path"], []).append(r)
    return grouped


# ── Mask colorisation helper ─────────────────────────────────────────────────

def colorize_mask(mask_u8):
    """Convert a uint8 class-index mask to an RGB image for display."""
    h, w = mask_u8.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_val, rgb in MASK_COLORS.items():
        color_img[mask_u8 == cls_val] = rgb
    return color_img


# ── Main processing ──────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PPE MASK GENERATOR")
    print(f"  Mask output dir : {MASK_DIR}")
    print(f"  Index CSV       : {os.path.join(OUT_DIR, 'mask_index.csv')}")
    print("=" * 65)

    # 1. Parse all annotations
    print("\n[1/4] Parsing annotations …")
    records = parse_minhnkb_annotations() + parse_jomarkow_annotations()
    print(f"  Total annotations : {len(records)}")
    grouped = group_by_image(records)
    total   = len(grouped)
    print(f"  Unique images     : {total}")

    # 2. Load SAM2 model
    print("\n[2/4] Loading SAM2 model (sam2_b.pt) …")
    try:
        from ultralytics import SAM
        sam_model = SAM("sam2_b.pt")
    except Exception as e:
        print(f"  ERROR loading SAM2: {e}")
        sys.exit(1)

    # 3. Generate masks
    print("\n[3/4] Generating masks …")
    index_rows   = []   # (img_path, mask_path, img_w, img_h, n_boxes)
    skipped      = 0
    processed    = 0

    for img_path, anns in grouped.items():
        stem      = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, f"{stem}_mask.png")

        img_w = anns[0]["img_w"]
        img_h = anns[0]["img_h"]
        boxes = [[a["x1"], a["y1"], a["x2"], a["y2"]] for a in anns]

        try:
            results = sam_model(img_path, bboxes=boxes)

            # Guard against None masks
            if results[0].masks is None:
                raise ValueError(f"SAM2 returned no masks for {img_path}")

            # results[0].masks.data → (N, H, W) bool tensor (CPU)
            masks_tensor = results[0].masks.data.cpu().numpy().astype(bool)
            n_masks = min(masks_tensor.shape[0], len(anns))

            # Initialise as 255 (background / ignore index)
            composite = np.full((img_h, img_w), 255, dtype=np.uint8)

            # Composite masks in annotation order; later boxes override earlier ones
            for i in range(n_masks):
                cls_idx = anns[i]["cls_idx"]
                m = masks_tensor[i].astype(bool)

                # SAM may return masks at model resolution; resize if necessary
                if m.shape != (img_h, img_w):
                    m = cv2.resize(
                        m.astype(np.uint8), (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                composite[m] = cls_idx

            cv2.imwrite(mask_path, composite)
            index_rows.append((img_path, mask_path, img_w, img_h, len(boxes)))
            processed += 1

        except Exception as exc:
            print(f"  WARNING: skipping {os.path.basename(img_path)} — {exc}")
            skipped += 1

        if (processed + skipped) % 50 == 0 and (processed + skipped) > 0:
            print(f"  [{processed + skipped}/{total}] processed so far, {skipped} skipped")

    print(f"\n  Done. Processed={processed}, Skipped={skipped}")

    # 4. Write mask_index.csv
    index_csv = os.path.join(OUT_DIR, "mask_index.csv")
    with open(index_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["img_path", "mask_path", "img_w", "img_h", "n_boxes"])
        writer.writerows(index_rows)
    print(f"  Index saved → {index_csv}  ({len(index_rows)} rows)")

    # 5. Verification montage
    print("\n[4/4] Creating verification montage …")
    if len(index_rows) == 0:
        print("  No masks generated — skipping montage.")
        return

    sample_rows = random.sample(index_rows, min(12, len(index_rows)))
    n_pairs = len(sample_rows)
    grid_cols = min(n_pairs, 6) * 2   # up to 6 pairs → 12 sub-images wide
    grid_rows = (n_pairs + 5) // 6    # ceiling(n_pairs / 6) rows of pairs

    fig, axes = plt.subplots(
        grid_rows * 1, grid_cols,
        figsize=(grid_cols * 2.5, grid_rows * 2.5),
        squeeze=False,
    )
    # Flatten axes for easy indexing; we use pairs: col 2k = image, col 2k+1 = mask
    # Layout: each row has up to 6 side-by-side (image, mask) pairs.

    col_per_row = min(n_pairs, 6)
    for idx, (img_path, mask_path, img_w, img_h, n_boxes) in enumerate(sample_rows):
        row_i  = idx // 6
        col_i  = (idx % 6) * 2
        ax_img = axes[row_i][col_i]
        ax_msk = axes[row_i][col_i + 1]

        # Load original image (RGB for display)
        orig = cv2.imread(img_path)
        if orig is not None:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        else:
            orig = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # Load and colorize mask
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            mask_raw = np.full((img_h, img_w), 255, dtype=np.uint8)
        mask_color = colorize_mask(mask_raw)

        ax_img.imshow(orig)
        ax_img.set_title(os.path.basename(img_path)[:18], fontsize=6)
        ax_img.axis("off")

        ax_msk.imshow(mask_color)
        ax_msk.set_title(f"mask  boxes={n_boxes}", fontsize=6)
        ax_msk.axis("off")

    # Hide unused axes
    total_ax_cols = grid_cols
    used_pairs = n_pairs
    for idx in range(used_pairs, grid_rows * 6):
        row_i = idx // 6
        col_i = (idx % 6) * 2
        if row_i < grid_rows and col_i < total_ax_cols:
            axes[row_i][col_i].axis("off")
        if row_i < grid_rows and col_i + 1 < total_ax_cols:
            axes[row_i][col_i + 1].axis("off")

    plt.suptitle("SAM2 Mask Verification — PPE Dataset", fontsize=10, y=1.01)
    plt.tight_layout()
    montage_path = os.path.join(OUT_DIR, "mask_verification.png")
    plt.savefig(montage_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Montage saved → {montage_path}")

    print("\n" + "=" * 65)
    print("PPE MASK GENERATOR — COMPLETE")
    print(f"  Masks written   : {processed}")
    print(f"  Skipped (error) : {skipped}")
    print(f"  Index CSV       : {index_csv}")
    print(f"  Montage         : {montage_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
