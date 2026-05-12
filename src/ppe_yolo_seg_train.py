"""
YOLOv8n-seg Instance Segmentation — Keremberke 10-class schema
===============================================================
Fine-tunes YOLOv8n-seg on the keremberke dataset built by
ppe_seg_keremberke_rebuild.py.

Classes (YOLO 0-indexed):
  0 = helmet    1 = no_helmet
  2 = glove     3 = no_glove
  4 = goggles   5 = no_goggles
  6 = mask      7 = no_mask
  8 = shoes     9 = no_shoes

Outputs:
  runs/segment/ppe_ke_seg/        YOLO training artefacts
  results/models/yolo_seg_results.csv
  results/models/yolo_seg_confusion.png
  results/models/yolo_seg_results_plot.png
"""

import os
import csv
import shutil
import time
from pathlib import Path

from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
for _cand in ["D:/datasets/ppe_seg_ke", "D:/Claude/datasets/ppe_seg_ke",
              str(Path(__file__).resolve().parents[2] / "datasets" / "ppe_seg_ke")]:
    yaml_cand = os.path.join(_cand, "instance", "ppe_seg_ke.yaml")
    if os.path.exists(yaml_cand):
        YAML_PATH = yaml_cand
        break
else:
    raise FileNotFoundError("ppe_seg_ke.yaml not found — run ppe_seg_keremberke_rebuild.py first")

PROJECT_DIR = str(Path(__file__).resolve().parent.parent)
OUT_DIR     = os.path.join(PROJECT_DIR, "results", "models")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = [
    "helmet",   "no_helmet",
    "glove",    "no_glove",
    "goggles",  "no_goggles",
    "mask",     "no_mask",
    "shoes",    "no_shoes",
]


def main():
    EPOCHS = 30
    IMGSZ  = 640
    BATCH  = 16
    NAME   = "ppe_ke_seg"

    print("=" * 65)
    print("YOLOv8n-seg — KEREMBERKE 10-CLASS INSTANCE SEGMENTATION")
    print(f"Data: {YAML_PATH}")
    print(f"Epochs: {EPOCHS}  imgsz: {IMGSZ}  batch: {BATCH}")
    print("=" * 65)

    model = YOLO("yolov8n-seg.pt")

    t0 = time.time()
    results = model.train(
        data    = YAML_PATH,
        epochs  = EPOCHS,
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = 0,
        name    = NAME,
        project = os.path.join(PROJECT_DIR, "runs", "segment"),
        exist_ok= True,
        verbose = True,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f}m")

    # ── Extract final metrics ──────────────────────────────────────────────
    run_dir = os.path.join(PROJECT_DIR, "runs", "segment", NAME)

    # Copy confusion matrix and results plot if generated
    for fname, dest in [
        ("confusion_matrix.png",       "yolo_seg_confusion.png"),
        ("results.png",                "yolo_seg_results_plot.png"),
    ]:
        src = os.path.join(run_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT_DIR, dest))
            print(f"Copied {fname} → results/models/{dest}")

    # Pull metrics from results object
    try:
        box_map50    = float(results.box.map50)
        box_map5095  = float(results.box.map)
        mask_map50   = float(results.seg.map50)
        mask_map5095 = float(results.seg.map)
        precision    = float(results.box.mp)
        recall       = float(results.box.mr)
    except Exception as e:
        print(f"Warning: could not extract metrics from results object ({e})")
        box_map50 = box_map5095 = mask_map50 = mask_map5095 = precision = recall = 0.0

    # Per-class metrics
    per_class_rows = []
    try:
        names = model.names
        for cls_id, cls_name in names.items():
            try:
                bp = float(results.box.p[cls_id])
                br = float(results.box.r[cls_id])
                bm = float(results.box.ap50[cls_id])
                mm = float(results.seg.ap50[cls_id])
                per_class_rows.append([cls_name, f"{bp:.4f}", f"{br:.4f}",
                                       f"{bm:.4f}", f"{mm:.4f}"])
            except Exception:
                per_class_rows.append([cls_name, "", "", "", ""])
    except Exception:
        pass

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "yolo_seg_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Task", "mask_mAP50", "box_mAP50",
                    "mask_mAP50_95", "box_mAP50_95",
                    "Precision", "Recall", "Train_Time(s)",
                    "Architecture", "Params_K", "Notes"])
        w.writerow([
            "YOLOv8n-seg (keremberke 10-class)", "instance_seg",
            f"{mask_map50:.3f}", f"{box_map50:.3f}",
            f"{mask_map5095:.3f}", f"{box_map5095:.3f}",
            f"{precision:.3f}", f"{recall:.3f}",
            f"{elapsed:.0f}", "YOLO", "3259",
            "10-class keremberke instance segmentation"
        ])
    print(f"Saved {csv_path}")

    # ── Per-class bar chart ────────────────────────────────────────────────
    if per_class_rows:
        names_pc  = [r[0] for r in per_class_rows]
        box_map50s= [float(r[3]) if r[3] else 0 for r in per_class_rows]
        msk_map50s= [float(r[4]) if r[4] else 0 for r in per_class_rows]

        x = np.arange(len(names_pc))
        fig, ax = plt.subplots(figsize=(13, 5))
        bars1 = ax.bar(x - 0.2, box_map50s, 0.4, label="Box mAP50",  color="#3498DB")
        bars2 = ax.bar(x + 0.2, msk_map50s, 0.4, label="Mask mAP50", color="#E74C3C")
        ax.axhline(box_map50,  color="#3498DB", linestyle="--", alpha=0.5,
                   label=f"Mean box mAP50={box_map50:.3f}")
        ax.axhline(mask_map50, color="#E74C3C", linestyle="--", alpha=0.5,
                   label=f"Mean mask mAP50={mask_map50:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(names_pc, rotation=30, ha="right")
        ax.set_ylim(0, 1.0); ax.set_title("YOLOv8n-seg Per-Class mAP50 (keremberke 10-class)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "yolo_seg_confusion.png"), dpi=150)
        plt.close()
        print("Saved yolo_seg_confusion.png (per-class mAP50 chart)")

    print(f"\nSummary:")
    print(f"  Box  mAP50     = {box_map50:.4f}")
    print(f"  Mask mAP50     = {mask_map50:.4f}")
    print(f"  Box  mAP50-95  = {box_map5095:.4f}")
    print(f"  Mask mAP50-95  = {mask_map5095:.4f}")


if __name__ == "__main__":
    main()
