"""
Dataset Setup Script
====================
Downloads and combines all datasets needed for PPE detection training.

Sources
-------
1. Jomarkow Hard Hat Workers (1,000 images + YOLO labels)
   GitHub: jomarkow/Safety-Helmet-Detection
   Classes: 0=helmet, 1=head (no helmet), 2=person
   Used for: person detection (head→body box expansion)

2. MinhNKB Helmet & Safety Vest Detection (1,500 images + XML annotations)
   GitHub: MinhNKB/helmet-safety-vest-detection
   Classes: helmet, safety_vest, no_ppe, partial_ppe, full_ppe (via class map)
   Used for: PPE crop classification training

3. INRIA Person Dataset (300 pedestrian crops)
   HuggingFace: marcelarosalesj/inria-person
   Used for: augmenting person detection training set

Outputs
-------
datasets/
├── jomarkow/
│   ├── images/          raw images (PNG)
│   └── labels/          YOLO txt labels
├── helmet-safety-vest-detection-master/
│   ├── train-images-data/       JPG images
│   └── train-images-annotations-new/  XML annotations
├── inria_person/
│   └── pedestrians/     cropped pedestrian PNGs
├── ppe_crops/           combined PPE classification dataset
│   ├── train/           80% split — subdirs per class
│   │   ├── helmet/
│   │   ├── safety_vest/
│   │   ├── no_ppe/
│   │   ├── partial_ppe/
│   │   └── full_ppe/
│   └── val/             20% split
└── person_detection/    YOLO format for person detector
    ├── train/images/
    ├── train/labels/
    ├── val/images/
    ├── val/labels/
    └── person_detect.yaml

Usage
-----
    python src/setup_datasets.py              # download + build everything
    python src/setup_datasets.py --skip-download  # rebuild from existing downloads
    python src/setup_datasets.py --only-download  # download only, no dataset build
"""

import os
import sys
import glob
import shutil
import random
import zipfile
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Repo-relative paths ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS  = REPO_ROOT / "datasets"

JOMARK_IMG  = DATASETS / "jomarkow" / "images"
JOMARK_LBL  = DATASETS / "jomarkow" / "labels"
MINHNKB_ROOT= DATASETS / "helmet-safety-vest-detection-master"
MINHNKB_IMG = MINHNKB_ROOT / "train-images-data"
MINHNKB_ANN = MINHNKB_ROOT / "train-images-annotations-new"
INRIA_DIR   = DATASETS / "inria_person" / "pedestrians"
PPE_CROPS   = DATASETS / "ppe_crops"
PERSON_DET  = DATASETS / "person_detection"

CLASSES     = ["helmet", "safety_vest", "no_ppe", "partial_ppe", "full_ppe"]
CLASS_IDX   = {c: i for i, c in enumerate(CLASSES)}

# ── Download helpers ─────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a single URL to dest. Returns True on success."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  FAIL {desc or url}: {e}")
        return False


def parallel_download(tasks: list[tuple[str, Path]], workers: int = 12, label: str = "") -> int:
    """Download (url, path) pairs in parallel. Returns success count."""
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_file, url, path): path for url, path in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                ok += 1
            if i % 100 == 0 or i == len(tasks):
                print(f"  {label}: {i}/{len(tasks)} ({ok} ok)", flush=True)
    return ok


# ── 1. Jomarkow ──────────────────────────────────────────────────────────────

JOMARK_BASE_IMG = "https://raw.githubusercontent.com/jomarkow/Safety-Helmet-Detection/main/data/images"
JOMARK_BASE_LBL = "https://raw.githubusercontent.com/jomarkow/Safety-Helmet-Detection/main/data/labels"
JOMARK_N        = 1000   # dataset has hard_hat_workers0 … hard_hat_workers999


def download_jomarkow():
    print("\n[1/3] Jomarkow Hard Hat Workers dataset")
    JOMARK_IMG.mkdir(parents=True, exist_ok=True)
    JOMARK_LBL.mkdir(parents=True, exist_ok=True)

    existing_imgs = len(list(JOMARK_IMG.glob("*.png")))
    existing_lbls = len(list(JOMARK_LBL.glob("*.txt")))
    if existing_imgs >= JOMARK_N and existing_lbls >= JOMARK_N:
        print(f"  Already have {existing_imgs} images + {existing_lbls} labels — skipping.")
        return

    stems = [f"hard_hat_workers{i}" for i in range(JOMARK_N)]

    img_tasks = [
        (f"{JOMARK_BASE_IMG}/{s}.png", JOMARK_IMG / f"{s}.png")
        for s in stems if not (JOMARK_IMG / f"{s}.png").exists()
    ]
    lbl_tasks = [
        (f"{JOMARK_BASE_LBL}/{s}.txt", JOMARK_LBL / f"{s}.txt")
        for s in stems if not (JOMARK_LBL / f"{s}.txt").exists()
    ]

    print(f"  Downloading {len(img_tasks)} images…")
    parallel_download(img_tasks, label="images")
    print(f"  Downloading {len(lbl_tasks)} labels…")
    parallel_download(lbl_tasks, label="labels")

    imgs = len(list(JOMARK_IMG.glob("*.png")))
    lbls = len(list(JOMARK_LBL.glob("*.txt")))
    print(f"  Done: {imgs} images, {lbls} labels")


# ── 2. MinhNKB ───────────────────────────────────────────────────────────────

MINHNKB_URL = (
    "https://github.com/MinhNKB/helmet-safety-vest-detection/archive/refs/heads/master.zip"
)


def download_minhnkb():
    print("\n[2/3] MinhNKB Helmet & Safety Vest Detection dataset")

    if MINHNKB_IMG.exists() and len(list(MINHNKB_IMG.glob("*.jpg"))) > 100:
        print(f"  Already extracted — skipping.")
        return

    zip_path = DATASETS / "helmet-vest-detection.zip"
    if not zip_path.exists():
        print(f"  Downloading archive (~50 MB)…")
        ok = download_file(MINHNKB_URL, zip_path, "MinhNKB zip")
        if not ok:
            print("  ERROR: could not download MinhNKB dataset.")
            return

    print(f"  Extracting…")
    DATASETS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATASETS)

    # GitHub archives extract as "<repo>-<branch>/"
    extracted = DATASETS / "helmet-safety-vest-detection-master"
    if extracted.exists():
        print(f"  Extracted to: {extracted}")
    else:
        print(f"  WARNING: Expected folder not found after extraction.")

    imgs = len(list(MINHNKB_IMG.glob("*.jpg"))) if MINHNKB_IMG.exists() else 0
    anns = len(list(MINHNKB_ANN.glob("*.xml"))) if MINHNKB_ANN.exists() else 0
    print(f"  Done: {imgs} images, {anns} XML annotations")


# ── 3. INRIA Person Dataset ──────────────────────────────────────────────────

def download_inria(n: int = 300):
    print(f"\n[3/3] INRIA Person Dataset ({n} crops from HuggingFace)")
    INRIA_DIR.mkdir(parents=True, exist_ok=True)

    existing = len(list(INRIA_DIR.glob("*.png")) + list(INRIA_DIR.glob("*.jpg")))
    if existing >= n:
        print(f"  Already have {existing} crops — skipping.")
        return

    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        print("  Installing huggingface_hub…")
        os.system("pip install huggingface_hub --quiet --break-system-packages")
        from huggingface_hub import list_repo_files, hf_hub_download

    raw_dir = DATASETS / "inria_person_raw"
    files = list(list_repo_files("marcelarosalesj/inria-person", repo_type="dataset"))
    ped_files = [f for f in files if "/pedestrians/" in f][:n]
    print(f"  Found {len(ped_files)} pedestrian crops — downloading…")

    def _dl(fname):
        try:
            local = hf_hub_download(
                repo_id="marcelarosalesj/inria-person",
                filename=fname,
                repo_type="dataset",
                local_dir=str(raw_dir),
            )
            dest = INRIA_DIR / Path(fname).name
            shutil.copy(local, dest)
            return True
        except Exception:
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_dl, f): f for f in ped_files}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                ok += 1
            if i % 50 == 0 or i == len(ped_files):
                print(f"  {i}/{len(ped_files)} ({ok} ok)", flush=True)

    print(f"  Done: {ok}/{len(ped_files)} downloaded")


# ── 4. Build PPE Crop Dataset ────────────────────────────────────────────────

# Maps MinhNKB XML class names → our 5-class scheme
MINHNKB_CLASS_MAP = {
    # actual names found in train-images-annotations-new/
    "helmet":                       "helmet",
    "safety vest":                  "safety_vest",
    "person with full safety":      "full_ppe",
    "person with partial safety":   "partial_ppe",
    "person without safety":        "no_ppe",
    # fallback aliases (other forks of the dataset)
    "head":                         "no_ppe",
    "person":                       "no_ppe",
    "safety_vest":                  "safety_vest",
    "vest":                         "safety_vest",
    "no_helmet":                    "no_ppe",
    "helmet_vest":                  "full_ppe",
    "helmet_novest":                "partial_ppe",
    "nohelmet_vest":                "partial_ppe",
    "nohelmet_novest":              "no_ppe",
}


def parse_minhnkb_xml(xml_path: Path):
    """Parse a Pascal VOC-style XML and return list of (class_name, x1, y1, x2, y2)."""
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        raw  = obj.findtext("name", "").strip().lower()
        # try exact match first, then underscore-normalised
        cls  = MINHNKB_CLASS_MAP.get(raw) or MINHNKB_CLASS_MAP.get(raw.replace(" ", "_"))
        if cls is None:
            continue
        bnd  = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            x1 = int(float(bnd.findtext("xmin")))
            y1 = int(float(bnd.findtext("ymin")))
            x2 = int(float(bnd.findtext("xmax")))
            y2 = int(float(bnd.findtext("ymax")))
        except (TypeError, ValueError):
            continue
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def build_ppe_crops(val_split: float = 0.2, target_size: int = 64):
    """Crop PPE regions from MinhNKB images and organise into train/val splits."""
    import cv2

    print("\n[4/4] Building combined PPE crop dataset")

    for split in ["train", "val"]:
        for cls in CLASSES:
            (PPE_CROPS / split / cls).mkdir(parents=True, exist_ok=True)

    xml_files = sorted(MINHNKB_ANN.glob("*.xml"))
    if not xml_files:
        print("  ERROR: No XML annotations found. Run download first.")
        return

    random.seed(42)
    records = []   # (img_path, cls, x1, y1, x2, y2)

    for xf in xml_files:
        stem = xf.stem
        img_path = MINHNKB_IMG / f"{stem}.jpg"
        if not img_path.exists():
            img_path = MINHNKB_IMG / f"{stem}.png"
        if not img_path.exists():
            continue
        for cls, x1, y1, x2, y2 in parse_minhnkb_xml(xf):
            records.append((img_path, cls, x1, y1, x2, y2))

    random.shuffle(records)
    n_val = int(len(records) * val_split)
    val_set  = set(range(n_val))

    counts = {c: {"train": 0, "val": 0} for c in CLASSES}
    skipped = 0

    for i, (img_path, cls, x1, y1, x2, y2) in enumerate(records):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            skipped += 1
            continue

        crop = img[y1c:y2c, x1c:x2c]
        crop = cv2.resize(crop, (target_size, target_size))

        split = "val" if i in val_set else "train"
        n     = counts[cls][split]
        out   = PPE_CROPS / split / cls / f"{stem}_{n:04d}.jpg"
        cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts[cls][split] += 1

    print(f"  Skipped {skipped} crops (missing images or bad boxes)")
    print(f"  {'Class':<15} {'Train':>6} {'Val':>6}")
    print(f"  {'-'*28}")
    total_train = total_val = 0
    for cls in CLASSES:
        t, v = counts[cls]["train"], counts[cls]["val"]
        total_train += t
        total_val   += v
        print(f"  {cls:<15} {t:>6} {v:>6}")
    print(f"  {'TOTAL':<15} {total_train:>6} {total_val:>6}")


# ── 5. Build Person Detection Dataset ────────────────────────────────────────

def build_person_detection(val_split: float = 0.15):
    """
    Build YOLO-format person detection dataset from:
    - Jomarkow: head/helmet boxes expanded to full-body person boxes
    - INRIA: full-image boxes (entire crop = one person)
    """
    import cv2

    print("\n[5/5] Building YOLO person detection dataset")

    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            (PERSON_DET / split / sub).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    samples = []  # (img_src_path, yolo_label_lines, stem)

    # ── Jomarkow: head → body expansion ──────────────────────────────────────
    lbl_files = sorted(JOMARK_LBL.glob("*.txt"))
    for lf in lbl_files:
        stem = lf.stem
        img_p = JOMARK_IMG / f"{stem}.png"
        if not img_p.exists():
            continue
        boxes = []
        for line in lf.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            if cls_id == 2:
                # explicit person box — use as-is
                boxes.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            elif cls_id in (0, 1):
                # head/helmet: expand to full body
                body_h  = min(bh * 7.5, 1.0)
                new_bw  = min(bw * 2.5, 1.0)
                new_cy  = cy + bh * 3.0
                new_cy  = min(max(new_cy, body_h / 2), 1.0 - body_h / 2)
                boxes.append(f"0 {cx:.6f} {new_cy:.6f} {new_bw:.6f} {body_h:.6f}")

        if boxes:
            samples.append((img_p, boxes, f"jomark_{stem}"))

    print(f"  Jomarkow: {len(samples)} images with person boxes")

    # ── INRIA: whole-image boxes ──────────────────────────────────────────────
    inria_imgs = list(INRIA_DIR.glob("*.png")) + list(INRIA_DIR.glob("*.jpg"))
    margin = 0.05
    for i, ip in enumerate(inria_imgs):
        # Full image = one person; add a small margin
        x1n, y1n = margin, margin
        x2n, y2n = 1 - margin, 1 - margin
        cx = (x1n + x2n) / 2
        cy = (y1n + y2n) / 2
        bw = x2n - x1n
        bh = y2n - y1n
        samples.append((ip, [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"], f"inria_{i:04d}"))

    print(f"  INRIA: {len(inria_imgs)} pedestrian crops added")
    print(f"  Total: {len(samples)} samples")

    random.shuffle(samples)
    n_val = int(len(samples) * val_split)

    copied = {"train": 0, "val": 0}
    for i, (src, label_lines, stem) in enumerate(samples):
        split = "val" if i < n_val else "train"
        dst_img = PERSON_DET / split / "images" / f"{stem}{src.suffix}"
        dst_lbl = PERSON_DET / split / "labels" / f"{stem}.txt"
        shutil.copy(src, dst_img)
        dst_lbl.write_text("\n".join(label_lines) + "\n")
        copied[split] += 1

    print(f"  Train: {copied['train']}  Val: {copied['val']}")

    # Write YAML config
    yaml_path = PERSON_DET / "person_detect.yaml"
    yaml_path.write_text(
        f"path: {PERSON_DET}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
    )
    print(f"  Config written: {yaml_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download and prepare PPE detection datasets")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloads, rebuild datasets from existing files")
    parser.add_argument("--only-download", action="store_true",
                        help="Download only, skip dataset build steps")
    parser.add_argument("--inria-n", type=int, default=300,
                        help="Number of INRIA crops to download (default: 300)")
    parser.add_argument("--crop-size", type=int, default=64,
                        help="PPE crop resize dimension (default: 64)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation fraction (default: 0.2)")
    args = parser.parse_args()

    DATASETS.mkdir(parents=True, exist_ok=True)
    print(f"Dataset root: {DATASETS}")

    if not args.skip_download:
        download_jomarkow()
        download_minhnkb()
        download_inria(n=args.inria_n)

    if not args.only_download:
        build_ppe_crops(val_split=args.val_split, target_size=args.crop_size)
        build_person_detection(val_split=args.val_split)

    print("\nDone. Dataset summary:")
    for d in [JOMARK_IMG, MINHNKB_IMG, INRIA_DIR]:
        if d.exists():
            n = sum(1 for _ in d.iterdir())
            print(f"  {d.relative_to(REPO_ROOT)}: {n} files")
    for split in ["train", "val"]:
        d = PPE_CROPS / split
        if d.exists():
            n = sum(1 for _ in d.rglob("*.jpg"))
            print(f"  ppe_crops/{split}: {n} crops")
    for split in ["train", "val"]:
        d = PERSON_DET / split / "images"
        if d.exists():
            n = sum(1 for _ in d.iterdir())
            print(f"  person_detection/{split}: {n} images")


if __name__ == "__main__":
    main()
