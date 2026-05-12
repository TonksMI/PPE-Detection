"""
Setup Script for End-to-End YOLO PPE Detection
Parses MinhNKB and Jomarkow datasets into a unified YOLO format
Using all available data.
"""
import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths — check known locations in priority order
for _candidate in ["D:/datasets", "D:/Claude/datasets",
                    os.path.join(os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__)))), "datasets")]:
    if os.path.exists(os.path.join(_candidate, "jomarkow")):
        DATASETS = _candidate
        break
else:
    DATASETS = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "datasets")

MINHNKB_IMG = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-data")
MINHNKB_ANN = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-annotations-new")
JOMARK_IMG  = os.path.join(DATASETS, "jomarkow/images")
JOMARK_LBL  = os.path.join(DATASETS, "jomarkow/labels")

YOLO_OUT = os.path.join(DATASETS, "yolo_ppe_end2end")

# Classes and mapping
CLASSES = ["helmet", "safety_vest", "full_ppe", "partial_ppe", "no_ppe"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

MINHNKB_MAP = {
    "helmet": "helmet",
    "safety vest": "safety_vest",
    "person with full safety": "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety": "no_ppe",
}
# Fallbacks
MINHNKB_MAP.update({
    "head": "no_ppe", "person": "no_ppe", "safety_vest": "safety_vest",
    "vest": "safety_vest", "no_helmet": "no_ppe", "helmet_vest": "full_ppe",
    "helmet_novest": "partial_ppe", "nohelmet_vest": "partial_ppe", "nohelmet_novest": "no_ppe"
})

JOMARK_MAP = {0: "helmet", 1: "no_ppe"}

def make_dirs():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(YOLO_OUT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(YOLO_OUT, split, 'labels'), exist_ok=True)

def process_minhnkb():
    print("Processing MinhNKB...")
    xml_files = sorted(glob.glob(os.path.join(MINHNKB_ANN, "*.xml")))
    records = []
    skipped = 0
    
    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            fname = root.findtext("filename")
            
            ip = os.path.join(MINHNKB_IMG, fname) if fname else None
            if not ip or not os.path.exists(ip):
                stem = os.path.splitext(os.path.basename(xf))[0]
                for ext in [".jpg", ".jpeg", ".png"]:
                    cand = os.path.join(MINHNKB_IMG, stem + ext)
                    if os.path.exists(cand):
                        ip = cand
                        break
            
            if not ip or not os.path.exists(ip):
                skipped += 1
                continue
                
            sz = root.find("size")
            iw = int(sz.findtext("width", 0))
            ih = int(sz.findtext("height", 0))
            if iw == 0 or ih == 0:
                # Try to get from image
                import cv2
                img = cv2.imread(ip)
                if img is None: continue
                ih, iw = img.shape[:2]
                
            yolo_lines = []
            for obj in root.findall("object"):
                raw = obj.findtext("name", "").strip().lower()
                cls = MINHNKB_MAP.get(raw) or MINHNKB_MAP.get(raw.replace(" ", "_"))
                if not cls: continue
                
                bb = obj.find("bndbox")
                x1 = float(bb.findtext("xmin"))
                y1 = float(bb.findtext("ymin"))
                x2 = float(bb.findtext("xmax"))
                y2 = float(bb.findtext("ymax"))
                
                # YOLO format: cls cx cy w h (normalized)
                cx = ((x1 + x2) / 2.0) / iw
                cy = ((y1 + y2) / 2.0) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                
                if bw > 0 and bh > 0:
                    yolo_lines.append(f"{CLASS_IDX[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            if yolo_lines:
                records.append((ip, yolo_lines))
        except Exception as e:
            skipped += 1
            pass
            
    print(f"MinhNKB parsed: {len(records)} images with labels (skipped {skipped})")
    return records

def process_jomarkow():
    print("Processing Jomarkow...")
    lbl_files = sorted(glob.glob(os.path.join(JOMARK_LBL, "*.txt")))
    records = []
    
    for lf in lbl_files:
        stem = os.path.splitext(os.path.basename(lf))[0]
        ip = os.path.join(JOMARK_IMG, stem + ".png")
        if not os.path.exists(ip):
            ip = os.path.join(JOMARK_IMG, stem + ".jpg")
            
        if not os.path.exists(ip): continue
        
        yolo_lines = []
        with open(lf, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls_id = int(parts[0])
                if cls_id not in JOMARK_MAP: continue
                
                cls = JOMARK_MAP[cls_id]
                # Jomarkow is already in YOLO format
                cx, cy, bw, bh = map(float, parts[1:5])
                
                yolo_lines.append(f"{CLASS_IDX[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
        if yolo_lines:
            records.append((ip, yolo_lines))
            
    print(f"Jomarkow parsed: {len(records)} images with labels")
    return records

def main():
    make_dirs()
    all_records = process_minhnkb() + process_jomarkow()
    random.seed(42)
    random.shuffle(all_records)
    
    val_split = 0.2
    n_val = int(len(all_records) * val_split)
    
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]
    
    def write_split(records, split):
        print(f"Writing {split} set: {len(records)} images")
        for i, (ip, lines) in enumerate(records):
            stem = os.path.splitext(os.path.basename(ip))[0]
            new_name = f"{split}_{i:05d}"
            
            # Copy image
            ext = os.path.splitext(ip)[1]
            dest_img = os.path.join(YOLO_OUT, split, 'images', new_name + ext)
            shutil.copy(ip, dest_img)
            
            # Write labels
            dest_lbl = os.path.join(YOLO_OUT, split, 'labels', new_name + ".txt")
            with open(dest_lbl, 'w') as f:
                f.write("\n".join(lines) + "\n")
                
    write_split(train_records, 'train')
    write_split(val_records, 'val')
    
    # Write YAML
    yaml_path = os.path.join(YOLO_OUT, "ppe_end2end.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {YOLO_OUT}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
            
    print(f"Done. Configuration written to: {yaml_path}")

if __name__ == "__main__":
    main()
