"""
PPE Detection — CCTV Validation & Model Performance Report Generator
Produces: results/plots/cctv_report.png (multi-page figure)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR  = os.path.join(PROJECT_DIR, "results", "models")
PLOTS_DIR   = os.path.join(PROJECT_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
cctv_csv   = os.path.join(MODELS_DIR, "prod_cctv_results.csv")
model_csv  = os.path.join(MODELS_DIR, "prod_model_summary.csv")
cctv_png   = os.path.join(MODELS_DIR, "prod_cctv_validation.png")

cctv_df  = pd.read_csv(cctv_csv)
model_df = pd.read_csv(model_csv)

# De-duplicate cctv rows (the CSV has duplicate rows)
cctv_df = cctv_df.drop_duplicates().reset_index(drop=True)

print("CCTV data:\n", cctv_df)
print("\nModel summary:\n", model_df)

# ── Colour palette ─────────────────────────────────────────────────────────
BRAND_BLUE  = '#1A3A5C'
BRAND_LIGHT = '#2C5F8A'
ACCENT      = '#27ae60'
WARN_RED    = '#e74c3c'
WARN_ORANGE = '#e67e22'
GRID_GRAY   = '#CCCCCC'
BG_LIGHT    = '#F4F8FC'

MODEL_PALETTE = [
    '#2980b9', '#27ae60', '#8e44ad',
    '#e67e22', '#e74c3c', '#1abc9c',
    '#f39c12', '#2c3e50', '#95a5a6',
]

# ── Helper: draw a section header bar ─────────────────────────────────────
def section_header(ax, title, subtitle=''):
    ax.set_facecolor(BRAND_BLUE)
    ax.text(0.02, 0.55, title, transform=ax.transAxes,
            fontsize=14, fontweight='bold', color='white', va='center')
    if subtitle:
        ax.text(0.02, 0.15, subtitle, transform=ax.transAxes,
                fontsize=9, color='#AADDFF', va='center')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Title + CCTV Validation Results
# ══════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(14, 18), facecolor='white')
fig1.patch.set_facecolor('white')

gs1 = gridspec.GridSpec(
    5, 2,
    figure=fig1,
    height_ratios=[0.7, 0.15, 2.8, 0.15, 3.2],
    hspace=0.35, wspace=0.25,
    left=0.06, right=0.96, top=0.96, bottom=0.04
)

# ── Title banner ──────────────────────────────────────────────────────────
ax_title = fig1.add_subplot(gs1[0, :])
ax_title.set_facecolor(BRAND_BLUE)
ax_title.text(0.5, 0.72, 'PPE Detection — CCTV Validation & Model Performance',
              transform=ax_title.transAxes, ha='center', va='center',
              fontsize=18, fontweight='bold', color='white')
ax_title.text(0.5, 0.32,
              'Production Pipeline Report  |  PPENetFast + YOLOv8n  |  February 2026',
              transform=ax_title.transAxes, ha='center', va='center',
              fontsize=10, color='#AADDFF')
ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis('off')

# ── Section header: CCTV ─────────────────────────────────────────────────
ax_hdr1 = fig1.add_subplot(gs1[1, :])
section_header(ax_hdr1, '1. CCTV Validation Results',
               'Real-world surveillance images — YOLOv8n person detection + PPENetFast classification')

# ── CCTV validation image ─────────────────────────────────────────────────
ax_img = fig1.add_subplot(gs1[2, :])
if os.path.exists(cctv_png):
    img = Image.open(cctv_png)
    ax_img.imshow(np.array(img))
    ax_img.set_title('Figure 1: CCTV Two-Stage Detection — YOLOv8n bounding boxes + PPENetFast PPE labels',
                     fontsize=9, color='#555555', style='italic', pad=6)
else:
    ax_img.text(0.5, 0.5, 'prod_cctv_validation.png not found',
                ha='center', va='center', fontsize=12, color='gray')
ax_img.axis('off')

# ── Section header: per-image stats ──────────────────────────────────────
ax_hdr2 = fig1.add_subplot(gs1[3, :])
section_header(ax_hdr2, '2. Per-Image CCTV Statistics',
               'People detected, classified, and violations flagged per image')

# ── Per-image bar chart ───────────────────────────────────────────────────
ax_bar = fig1.add_subplot(gs1[4, 0])
ax_pie = fig1.add_subplot(gs1[4, 1])

# Filter out the huge crowd image (4.jpg) for clearer visualisation, show separately
crowd_mask = cctv_df['people'] < 50
normal_df  = cctv_df[crowd_mask].copy()
crowd_df   = cctv_df[~crowd_mask].copy()

x     = np.arange(len(normal_df))
width = 0.28

b1 = ax_bar.bar(x - width,   normal_df['people'],      width, label='People detected', color=BRAND_LIGHT, alpha=0.9)
b2 = ax_bar.bar(x,            normal_df['classified'],  width, label='Classified',       color=ACCENT,      alpha=0.9)
b3 = ax_bar.bar(x + width,   normal_df['violations'],  width, label='Violations',        color=WARN_RED,    alpha=0.9)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(normal_df['file'], rotation=40, ha='right', fontsize=8)
ax_bar.set_ylabel('Count', fontsize=9)
ax_bar.set_title('Per-image detection breakdown\n(standard scenes, excl. crowd image)',
                 fontsize=9, fontweight='bold')
ax_bar.legend(fontsize=8, loc='upper right')
ax_bar.yaxis.grid(True, color=GRID_GRAY, linewidth=0.5)
ax_bar.set_facecolor(BG_LIGHT)
ax_bar.set_axisbelow(True)

# Annotate bar tops
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        str(int(h)), ha='center', va='bottom', fontsize=7)

# Note about crowd image
if not crowd_df.empty:
    crowd_row = crowd_df.iloc[0]
    ax_bar.text(0.5, -0.35,
                f"Note: {crowd_row['file']} (crowd scene) — {int(crowd_row['people'])} people, "
                f"{int(crowd_row['classified'])} classified, {int(crowd_row['violations'])} violations — shown separately",
                transform=ax_bar.transAxes, ha='center', fontsize=7.5,
                color='#666666', style='italic')

# ── Violation vs compliant pie chart ──────────────────────────────────────
total_people    = int(cctv_df['people'].sum())
total_classified = int(cctv_df['classified'].sum())
total_violations = int(cctv_df['violations'].sum())
total_compliant  = total_classified - total_violations

sizes  = [total_compliant, total_violations,
          total_people - total_classified]
labels = [
    f'Compliant\n({total_compliant})',
    f'Violations\n({total_violations})',
    f'Unclassified\n({total_people - total_classified})',
]
colors  = [ACCENT, WARN_RED, '#BDC3C7']
explode = (0, 0.08, 0)

wedges, texts, autotexts = ax_pie.pie(
    sizes, labels=labels, colors=colors, explode=explode,
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 9},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')

ax_pie.set_title(
    f'PPE Compliance Overview\n(Total people across all scenes: {total_people})',
    fontsize=9, fontweight='bold'
)

plt.savefig(os.path.join(PLOTS_DIR, 'cctv_report_page1.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print("Saved: cctv_report_page1.png")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance Summary
# ══════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(14, 18), facecolor='white')
gs2 = gridspec.GridSpec(
    6, 2,
    figure=fig2,
    height_ratios=[0.55, 0.12, 2.2, 0.12, 2.2, 1.5],
    hspace=0.45, wspace=0.30,
    left=0.06, right=0.96, top=0.96, bottom=0.04
)

# ── Title ─────────────────────────────────────────────────────────────────
ax_t2 = fig2.add_subplot(gs2[0, :])
ax_t2.set_facecolor(BRAND_BLUE)
ax_t2.text(0.5, 0.72, 'PPE Detection — Model Performance Summary',
           transform=ax_t2.transAxes, ha='center', va='center',
           fontsize=18, fontweight='bold', color='white')
ax_t2.text(0.5, 0.28,
           'SVM / RF / ExtraTrees / HistGBM / Ensemble / CNN PPENet  |  3,626 crops, 5 classes',
           transform=ax_t2.transAxes, ha='center', va='center',
           fontsize=10, color='#AADDFF')
ax_t2.set_xlim(0, 1); ax_t2.set_ylim(0, 1); ax_t2.axis('off')

# ── Section: Multi-class models ───────────────────────────────────────────
ax_hm = fig2.add_subplot(gs2[1, :])
section_header(ax_hm, '3. Multi-Class Model Comparison (5 classes)',
               'helmet / safety_vest / no_ppe / partial_ppe / full_ppe')

multi_df = model_df[model_df['Task'] == 'multi'].copy().reset_index(drop=True)
binary_df = model_df[model_df['Task'] == 'binary'].copy().reset_index(drop=True)

def short_name(n):
    n = str(n)
    n = n.replace('ExtraTrees (400 trees, multi)', 'ExtraTrees')
    n = n.replace('RandomForest (400 trees, multi)', 'RandomForest')
    n = n.replace('HistGBM (400 rounds, multi)', 'HistGBM')
    n = n.replace('SVM (PCA->RBF, multi)', 'SVM')
    n = n.replace('Ensemble (SVM+RF+ET+GBM, multi)', 'Ensemble')
    n = n.replace('ExtraTrees (400 trees, binary)', 'ExtraTrees')
    n = n.replace('RandomForest (400 trees, binary)', 'RandomForest')
    n = n.replace('HistGBM (400 rounds, binary)', 'HistGBM')
    n = n.replace('SVM (PCA->RBF, binary)', 'SVM')
    return n

multi_df['ShortName']  = multi_df['Model'].apply(short_name)
binary_df['ShortName'] = binary_df['Model'].apply(short_name)

# ── Multi-class grouped bar ───────────────────────────────────────────────
ax_mc = fig2.add_subplot(gs2[2, :])
xm = np.arange(len(multi_df))
bw = 0.28
colors_mc = MODEL_PALETTE[:len(multi_df)]

bars_acc = ax_mc.bar(xm - bw/2, multi_df['Accuracy'],  bw, label='Accuracy',  color=BRAND_LIGHT, alpha=0.9)
bars_f1  = ax_mc.bar(xm + bw/2, multi_df['Macro F1'],  bw, label='Macro F1',  color=ACCENT,      alpha=0.9)

ax_mc.set_xticks(xm)
ax_mc.set_xticklabels(multi_df['ShortName'], rotation=25, ha='right', fontsize=9)
ax_mc.set_ylabel('Score', fontsize=9)
ax_mc.set_ylim(0, 1.05)
ax_mc.set_title('Multi-class Performance (5-way classification)', fontsize=10, fontweight='bold')
ax_mc.legend(fontsize=9, loc='lower right')
ax_mc.yaxis.grid(True, color=GRID_GRAY, linewidth=0.5)
ax_mc.set_facecolor(BG_LIGHT)
ax_mc.set_axisbelow(True)

for bar in bars_acc:
    h = bar.get_height()
    ax_mc.text(bar.get_x() + bar.get_width()/2, h + 0.005,
               f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar in bars_f1:
    h = bar.get_height()
    ax_mc.text(bar.get_x() + bar.get_width()/2, h + 0.005,
               f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Best model annotation
best_idx = multi_df['Accuracy'].idxmax()
best_acc = multi_df.loc[best_idx, 'Accuracy']
best_nm  = multi_df.loc[best_idx, 'ShortName']
ax_mc.axhline(best_acc, color=WARN_ORANGE, linewidth=1.2, linestyle='--', alpha=0.7)
ax_mc.text(len(multi_df) - 0.5, best_acc + 0.012,
           f'Best: {best_nm} ({best_acc:.1%})',
           color=WARN_ORANGE, fontsize=8, ha='right', fontweight='bold')

# ── Section: Binary models ────────────────────────────────────────────────
ax_hb = fig2.add_subplot(gs2[3, :])
section_header(ax_hb, '4. Binary Model Comparison (PPE vs No-PPE)',
               'SVM / RandomForest / ExtraTrees / HistGBM')

ax_bc = fig2.add_subplot(gs2[4, :])
xb = np.arange(len(binary_df))
bars_bacc = ax_bc.bar(xb - bw/2, binary_df['Accuracy'],  bw, label='Accuracy',  color=BRAND_LIGHT, alpha=0.9)
bars_bf1  = ax_bc.bar(xb + bw/2, binary_df['Macro F1'],  bw, label='Macro F1',  color=ACCENT,      alpha=0.9)

ax_bc.set_xticks(xb)
ax_bc.set_xticklabels(binary_df['ShortName'], rotation=20, ha='right', fontsize=9)
ax_bc.set_ylabel('Score', fontsize=9)
ax_bc.set_ylim(0, 1.05)
ax_bc.set_title('Binary Performance (PPE present vs absent)', fontsize=10, fontweight='bold')
ax_bc.legend(fontsize=9, loc='lower right')
ax_bc.yaxis.grid(True, color=GRID_GRAY, linewidth=0.5)
ax_bc.set_facecolor(BG_LIGHT)
ax_bc.set_axisbelow(True)

for bar in bars_bacc:
    h = bar.get_height()
    ax_bc.text(bar.get_x() + bar.get_width()/2, h + 0.005,
               f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar in bars_bf1:
    h = bar.get_height()
    ax_bc.text(bar.get_x() + bar.get_width()/2, h + 0.005,
               f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

best_bidx = binary_df['Accuracy'].idxmax()
best_bacc = binary_df.loc[best_bidx, 'Accuracy']
best_bnm  = binary_df.loc[best_bidx, 'ShortName']
ax_bc.axhline(best_bacc, color=WARN_ORANGE, linewidth=1.2, linestyle='--', alpha=0.7)
ax_bc.text(len(binary_df) - 0.5, best_bacc + 0.012,
           f'Best: {best_bnm} ({best_bacc:.1%})',
           color=WARN_ORANGE, fontsize=8, ha='right', fontweight='bold')

# ── Summary key-metrics table ─────────────────────────────────────────────
ax_tbl = fig2.add_subplot(gs2[5, :])
ax_tbl.axis('off')

# Build summary rows
all_models = pd.concat([multi_df, binary_df], ignore_index=True)
table_data = []
col_labels  = ['Model', 'Task', 'Accuracy', 'Macro F1', 'Weighted F1', 'Train Time (s)']

for _, row in all_models.iterrows():
    table_data.append([
        row['ShortName'],
        row['Task'].capitalize(),
        f"{row['Accuracy']:.4f}",
        f"{row['Macro F1']:.4f}",
        f"{row['Weighted F1']:.4f}",
        f"{row['Train Time(s)']:.1f}",
    ])

tbl = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.35)

# Style header
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor(BRAND_BLUE)
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')

# Style alternating rows; highlight best multi-class model
for i, (_, row) in enumerate(all_models.iterrows(), start=1):
    row_color = '#EEF4FB' if i % 2 == 0 else 'white'
    # Highlight CNN (best performer)
    if 'CNN' in str(row['Model']):
        row_color = '#D4EFDF'  # light green
    for j in range(len(col_labels)):
        tbl[(i, j)].set_facecolor(row_color)

ax_tbl.set_title('Table 1: Full Model Performance Summary\n(green = CNN PPENet — best multi-class model)',
                 fontsize=9, fontweight='bold', pad=6)

plt.savefig(os.path.join(PLOTS_DIR, 'cctv_report_page2.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print("Saved: cctv_report_page2.png")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Combined summary dashboard (single-page overview)
# ══════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(16, 10), facecolor='white')
gs3 = gridspec.GridSpec(2, 3, figure=fig3,
                         height_ratios=[1, 1],
                         hspace=0.50, wspace=0.35,
                         left=0.06, right=0.96, top=0.90, bottom=0.08)

fig3.suptitle('PPE Detection — CCTV Validation & Model Performance Dashboard',
              fontsize=15, fontweight='bold', color=BRAND_BLUE, y=0.96)
fig3.text(0.5, 0.92, 'Production PPENetFast (87.05% val acc) + YOLOv8n | February 2026',
          ha='center', fontsize=9, color='#555555', style='italic')

# ── KPI boxes (row 0, spans all) ─────────────────────────────────────────
# We'll use the first row for 3 KPI panels + CCTV stats
ax_k1 = fig3.add_subplot(gs3[0, 0])
ax_k2 = fig3.add_subplot(gs3[0, 1])
ax_k3 = fig3.add_subplot(gs3[0, 2])

def kpi_box(ax, value, label, sub='', color=BRAND_BLUE):
    ax.set_facecolor(color)
    ax.text(0.5, 0.60, value, transform=ax.transAxes,
            ha='center', va='center', fontsize=26, fontweight='bold', color='white')
    ax.text(0.5, 0.25, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=11, color='#DDEEFF')
    if sub:
        ax.text(0.5, 0.08, sub, transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='#AADDFF', style='italic')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

best_multi_acc  = multi_df['Accuracy'].max()
best_binary_acc = binary_df['Accuracy'].max() if not binary_df.empty else 0.0
violation_rate  = total_violations / total_classified * 100 if total_classified > 0 else 0

kpi_box(ax_k1, f'{best_multi_acc:.1%}', 'Best Multi-class Accuracy',
        'CNN PPENetFast (30 epochs)', color='#1A5276')
kpi_box(ax_k2, f'{best_binary_acc:.1%}', 'Best Binary Accuracy',
        'SVM PCA→RBF (PPE vs no-PPE)', color='#145A32')
kpi_box(ax_k3, f'{violation_rate:.1f}%', 'CCTV Violation Rate',
        f'{total_violations} violations / {total_classified} classified', color='#7B241C')

# ── Row 1: multi-class accuracy bar, binary bar, CCTV pie ────────────────
ax_m = fig3.add_subplot(gs3[1, 0])
ax_b = fig3.add_subplot(gs3[1, 1])
ax_p = fig3.add_subplot(gs3[1, 2])

# Multi-class
bm = ax_m.barh(multi_df['ShortName'], multi_df['Accuracy'],
               color=MODEL_PALETTE[:len(multi_df)], alpha=0.88, edgecolor='white')
ax_m.set_xlim(0.5, 1.0)
ax_m.set_xlabel('Accuracy', fontsize=9)
ax_m.set_title('Multi-class Accuracy', fontsize=10, fontweight='bold')
ax_m.set_facecolor(BG_LIGHT)
ax_m.xaxis.grid(True, color=GRID_GRAY, linewidth=0.5)
ax_m.set_axisbelow(True)
for bar, val in zip(bm, multi_df['Accuracy']):
    ax_m.text(val + 0.002, bar.get_y() + bar.get_height()/2,
              f'{val:.3f}', va='center', fontsize=8, fontweight='bold')

# Binary
bb = ax_b.barh(binary_df['ShortName'], binary_df['Accuracy'],
               color=MODEL_PALETTE[:len(binary_df)], alpha=0.88, edgecolor='white')
ax_b.set_xlim(0.6, 1.0)
ax_b.set_xlabel('Accuracy', fontsize=9)
ax_b.set_title('Binary Accuracy (PPE vs No-PPE)', fontsize=10, fontweight='bold')
ax_b.set_facecolor(BG_LIGHT)
ax_b.xaxis.grid(True, color=GRID_GRAY, linewidth=0.5)
ax_b.set_axisbelow(True)
for bar, val in zip(bb, binary_df['Accuracy']):
    ax_b.text(val + 0.002, bar.get_y() + bar.get_height()/2,
              f'{val:.3f}', va='center', fontsize=8, fontweight='bold')

# CCTV pie
wsizes  = [total_compliant, total_violations, total_people - total_classified]
wlabels = [f'Compliant\n{total_compliant}', f'Violations\n{total_violations}',
           f'Unclassified\n{total_people - total_classified}']
wcolors = [ACCENT, WARN_RED, '#BDC3C7']
wedges2, texts2, auto2 = ax_p.pie(
    wsizes, labels=wlabels, colors=wcolors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    textprops={'fontsize': 8.5}
)
for at in auto2:
    at.set_fontsize(8.5)
    at.set_fontweight('bold')
ax_p.set_title(f'CCTV Compliance\n({total_people} people total)', fontsize=10, fontweight='bold')

plt.savefig(os.path.join(PLOTS_DIR, 'cctv_report_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print("Saved: cctv_report_dashboard.png")

# ── Print summary to console ───────────────────────────────────────────────
print("\n" + "=" * 62)
print("  PPE Detection — Report Summary")
print("=" * 62)
print(f"\n  CCTV Validation ({len(cctv_df)} unique scenes):")
print(f"    Total people detected   : {total_people}")
print(f"    Total classified        : {total_classified}")
print(f"    Total violations flagged: {total_violations}")
print(f"    Violation rate          : {violation_rate:.1f}%")

print(f"\n  Best Multi-class Model  : {multi_df.loc[multi_df['Accuracy'].idxmax(), 'ShortName']}")
print(f"    Accuracy  = {best_multi_acc:.4f}")
print(f"    Macro F1  = {multi_df['Macro F1'].max():.4f}")

print(f"\n  Best Binary Model       : {binary_df.loc[binary_df['Accuracy'].idxmax(), 'ShortName']}")
print(f"    Accuracy  = {best_binary_acc:.4f}")

print(f"\n  Output files saved to: {PLOTS_DIR}")
print("    - cctv_report_page1.png   (CCTV validation + per-image breakdown)")
print("    - cctv_report_page2.png   (model comparison charts + summary table)")
print("    - cctv_report_dashboard.png (single-page KPI dashboard)")
print("=" * 62)
