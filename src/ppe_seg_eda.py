"""
ppe_seg_eda.py
--------------
Exploratory Data Analysis for the keremberke PPE segmentation dataset,
formatted to match the Assignment 1 EDA figure (2×3 subplot grid).

Uses the YOLO-format instance labels (class cx cy w h + polygon points)
which exist for both train and val splits.

Output: results/plots/02_seg_eda_analysis.png
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent

for _cand in [
    "D:/Claude/datasets/ppe_seg_ke",
    "D:/datasets/ppe_seg_ke",
    str(BASE.parent / "datasets" / "ppe_seg_ke"),
]:
    if os.path.exists(os.path.join(_cand, "instance", "train", "labels")):
        SEG_ROOT = Path(_cand)
        break
else:
    raise FileNotFoundError("ppe_seg_ke not found — run src/ppe_seg_keremberke_rebuild.py first")

OUT_DIR = BASE / "results" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Class schema (10 PPE classes, no background — matches YOLO instance index) ──
CLASSES = [
    "helmet", "no_helmet", "glove", "no_glove",
    "goggles", "no_goggles", "mask", "no_mask", "shoes", "no_shoes",
]
N_CLS = len(CLASSES)

# Colour palette — one per class, presence = cool, absence = warm
COLORS = [
    "#27AE60",  # 0  helmet      green
    "#E74C3C",  # 1  no_helmet   red
    "#3498DB",  # 2  glove       blue
    "#9B59B6",  # 3  no_glove    purple
    "#F39C12",  # 4  goggles     orange
    "#E67E22",  # 5  no_goggles  dark orange
    "#1ABC9C",  # 6  mask        teal
    "#C0392B",  # 7  no_mask     dark red
    "#95A5A6",  # 8  shoes       grey
    "#2C3E50",  # 9  no_shoes    navy
]

PRESENCE_CLASSES  = [0, 2, 4, 6, 8]   # helmet, glove, goggles, mask, shoes
ABSENCE_CLASSES   = [1, 3, 5, 7, 9]   # no_helmet, no_glove, no_goggles, no_mask, no_shoes

# ── Parse all instance labels (train + val) ────────────────────────────────
print("Parsing instance labels …")

records = []   # list of dicts: split, img_stem, cls_idx, cx, cy, w, h

for split in ("train", "val"):
    lbl_dir = SEG_ROOT / "instance" / split / "labels"
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        stem = lbl_path.stem
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_idx = int(parts[0])
                if cls_idx >= N_CLS:
                    continue
                cx = float(parts[1])
                cy = float(parts[2])
                w  = float(parts[3])
                h  = float(parts[4])
                records.append({
                    "split":    split,
                    "img_stem": stem,
                    "cls_idx":  cls_idx,
                    "cls_name": CLASSES[cls_idx],
                    "cx":       cx,
                    "cy":       cy,
                    "w":        w,
                    "h":        h,
                    "box_area": w * h,          # fraction of image area
                    "aspect":   w / (h + 1e-6),
                })

print(f"  Total annotations : {len(records):,}")
print(f"  Unique images     : {len(set(r['img_stem'] for r in records)):,}")

# ── Aggregate statistics ────────────────────────────────────────────────────
class_counts = [0] * N_CLS
for r in records:
    class_counts[r["cls_idx"]] += 1

anns_per_image = {}
for r in records:
    anns_per_image.setdefault(r["img_stem"], 0)
    anns_per_image[r["img_stem"]] += 1

print("\n  Class distribution:")
print(f"  {'Class':<14} {'Count':>8}  {'%':>6}")
print(f"  {'-'*32}")
total = sum(class_counts)
for i, (cls, cnt) in enumerate(zip(CLASSES, class_counts)):
    print(f"  {cls:<14} {cnt:>8,}  {cnt/total*100:>5.1f}%")
print(f"  {'TOTAL':<14} {total:>8,}  100.0%")

# ── Figure — 2×3 grid, matching A1 EDA layout ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "PPE Segmentation — Exploratory Data Analysis\n"
    "keremberke dataset  |  10-class individual-item schema  |  4,000 images",
    fontsize=15, fontweight="bold"
)

# ── Panel 1 (0,0) — Class Distribution (Instance Annotations) ─────────────
ax = axes[0, 0]
bar_colors = [COLORS[i] for i in range(N_CLS)]
bars = ax.bar(CLASSES, class_counts, color=bar_colors, edgecolor="white", linewidth=0.8)
ax.set_title("Class Distribution (Instance Annotations)")
ax.set_xlabel("")
ax.set_ylabel("Annotation count")
ax.tick_params(axis="x", rotation=35)
for p, cnt in zip(bars, class_counts):
    ax.annotate(
        f"{cnt:,}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center", va="bottom", fontsize=7.5,
    )
ax.set_xticks(range(N_CLS))
ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── Panel 2 (0,1) — Presence vs Absence Pie ───────────────────────────────
ax = axes[0, 1]
presence_total = sum(class_counts[i] for i in PRESENCE_CLASSES)
absence_total  = sum(class_counts[i] for i in ABSENCE_CLASSES)
pie_vals   = [presence_total, absence_total]
pie_labels = [f"PPE present\n({presence_total:,})", f"PPE absent\n({absence_total:,})"]
pie_colors = ["#27AE60", "#E74C3C"]
wedges, texts, autotexts = ax.pie(
    pie_vals, labels=pie_labels, autopct="%1.1f%%",
    colors=pie_colors, startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 9},
)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title("Presence vs Absence Annotations")

# ── Panel 3 (0,2) — Log(Box Area) Distribution ────────────────────────────
ax = axes[0, 2]
# group records by class
for i, cls in enumerate(CLASSES):
    areas = [r["box_area"] for r in records if r["cls_idx"] == i]
    if areas:
        ax.hist(
            np.log1p(np.array(areas)),
            bins=35, alpha=0.55, label=cls,
            color=COLORS[i], edgecolor="none",
        )
ax.set_title("Log(Box Area) Distribution")
ax.set_xlabel("log(1 + bbox area as fraction of image)")
ax.set_ylabel("Count")
ax.legend(fontsize=6.5, ncol=2)
ax.grid(alpha=0.25)

# ── Panel 4 (1,0) — Coverage (bbox area / image area, already normalised) ─
ax = axes[1, 0]
for i, cls in enumerate(CLASSES):
    areas = [r["box_area"] for r in records if r["cls_idx"] == i]
    if areas:
        ax.hist(
            np.array(areas),
            bins=35, alpha=0.55, label=cls,
            color=COLORS[i], edgecolor="none",
        )
ax.set_title("Coverage (bbox area / image area)")
ax.set_xlabel("Coverage (0 – 1)")
ax.set_ylabel("Count")
ax.legend(fontsize=6.5, ncol=2)
ax.grid(alpha=0.25)

# ── Panel 5 (1,1) — Annotations per Image ─────────────────────────────────
ax = axes[1, 1]
api_vals = list(anns_per_image.values())
max_api  = max(api_vals)
ax.hist(
    api_vals,
    bins=range(1, min(max_api + 2, 35)),
    color="#3498DB", edgecolor="black", linewidth=0.6, align="left",
)
ax.set_title("Annotations per Image")
ax.set_xlabel("# annotations")
ax.set_ylabel("# images")
ax.axvline(np.mean(api_vals), color="crimson", linestyle="--", linewidth=1.4,
           label=f"mean = {np.mean(api_vals):.1f}")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# ── Panel 6 (1,2) — Aspect Ratio Distribution ─────────────────────────────
ax = axes[1, 2]
for i, cls in enumerate(CLASSES):
    aspects = [r["aspect"] for r in records if r["cls_idx"] == i]
    if aspects:
        clipped = np.clip(np.array(aspects), 0, 4)
        ax.hist(
            clipped,
            bins=35, alpha=0.55, label=cls,
            color=COLORS[i], edgecolor="none",
        )
ax.axvline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6, label="square (W=H)")
ax.set_title("Aspect Ratio Distribution (W/H, clipped 0–4)")
ax.set_xlabel("W / H")
ax.set_ylabel("Count")
ax.legend(fontsize=6.5, ncol=2)
ax.grid(alpha=0.25)

# ── Save ───────────────────────────────────────────────────────────────────
plt.tight_layout()
out_path = OUT_DIR / "02_seg_eda_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved -> {out_path}")

# ── Console summary ────────────────────────────────────────────────────────
presence_n = sum(anns_per_image[k] for k in anns_per_image)
print(f"\nSummary")
print(f"  Total annotations : {total:,}")
print(f"  Unique images     : {len(anns_per_image):,}")
print(f"  Avg. ann/image    : {np.mean(api_vals):.2f}  (median {np.median(api_vals):.0f})")
print(f"  Max ann/image     : {max_api}")
print(f"  Presence classes  : {presence_total:,}  ({presence_total/total*100:.1f}%)")
print(f"  Absence classes   : {absence_total:,}  ({absence_total/total*100:.1f}%)")
rarest = CLASSES[np.argmin(class_counts)]
common = CLASSES[np.argmax(class_counts)]
print(f"  Rarest class      : {rarest}  ({min(class_counts):,})")
print(f"  Most common class : {common}  ({max(class_counts):,})")
print(f"  Imbalance ratio   : {max(class_counts)/max(min(class_counts),1):.1f}×")
