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
            print(f"  keremberke: {processed}/{len(split_map_hf)} done ...")

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

    print(f"\n[1/3] MinhNKB ({len(xml_stems)} images, val={n_val}) ...")
    process_minhnkb(split_map)

    # ── keremberke HF download + SAM2 ─────────────────────────────────────
    print(f"\n[2/3] keremberke HF dataset (sampling {MAX_HF} images) ...")
    try:
        from datasets import load_dataset
        hf_ds = load_dataset("keremberke/protective-equipment-detection",
                              name="full")
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
        print(f"  WARNING: keremberke download failed ({e}) -- using MinhNKB only")

    # ── YOLO YAML ─────────────────────────────────────────────────────────
    print("\n[3/3] Writing YOLO YAML ...")
    yaml_path = write_yolo_yaml()

    # ── Summary ───────────────────────────────────────────────────────────
    for split in ("train", "val"):
        n = len(os.listdir(os.path.join(SEM_ROOT, split, "images")))
        print(f"  Semantic {split}: {n} images")
    print(f"\nDataset ready. YOLO config: {yaml_path}")


if __name__ == "__main__":
    main()
