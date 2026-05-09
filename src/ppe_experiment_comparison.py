"""
PPE EXPERIMENT COMPARISON
==========================
Read-only aggregation script. Loads all result CSVs produced by the
experiment scripts (ppe_early_stopping.py, ppe_rnn_train.py,
ppe_unet_train.py) plus the existing production summary CSVs, and
produces a unified academic comparison table + charts. Never retrains.

Outputs (all written to results/models/):
  experiment_comparison.png      — horizontal bar chart of accuracy
  experiment_f1_heatmap.png      — per-class F1 heatmap
  experiment_table.tex           — LaTeX tabular environment
  experiment_comparison_full.csv — full harmonised DataFrame
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BASE        = os.path.dirname(PROJECT_DIR)

if os.path.exists("D:/datasets/jomarkow"):
    DATASETS = "D:/datasets"
else:
    DATASETS = os.path.join(BASE, "datasets")

OUT_DIR = os.path.join(PROJECT_DIR, "results", "models")

ALL_CLASSES = ["full_ppe", "helmet", "no_ppe", "partial_ppe", "safety_vest"]

# Param counts in thousands
PARAM_COUNTS = {
    'CNN':        226,
    'GRU':        100,
    'LSTM':       800,
    'UNet':      7700,
    'SVM':          0,
    'RF':           0,
    'ExtraTrees':   0,
    'HistGBM':      0,
    'Ensemble':     0,
}

# Architecture colour palette (used across plots)
ARCH_COLOURS = {
    'CNN':        '#4C72B0',
    'GRU':        '#55A868',
    'LSTM':       '#C44E52',
    'UNet':       '#8172B2',
    'SVM':        '#CCB974',
    'RF':         '#64B5CD',
    'ExtraTrees': '#DD8452',
    'HistGBM':    '#937860',
    'Ensemble':   '#DA8BC3',
}

CNN_BASELINE = 0.8733  # production CNN accuracy


# ---------------------------------------------------------------------------
# Architecture inference helper
# ---------------------------------------------------------------------------
def _infer_arch(model_name: str) -> str:
    """Guess architecture tag from a model display name string."""
    mn = model_name.lower()
    if 'svm' in mn:
        return 'SVM'
    if 'randomforest' in mn or 'random forest' in mn or mn.startswith('rf'):
        return 'RF'
    if 'extratrees' in mn or 'extra trees' in mn or mn.startswith('et'):
        return 'ExtraTrees'
    if 'histgbm' in mn or 'hist' in mn or 'gbm' in mn:
        return 'HistGBM'
    if 'ensemble' in mn:
        return 'Ensemble'
    if 'gru' in mn or 'fastgru' in mn:
        return 'GRU'
    if 'lstm' in mn or 'normallstm' in mn:
        return 'LSTM'
    if 'unet' in mn or 'u-net' in mn:
        return 'UNet'
    if 'cnn' in mn or 'ppenet' in mn:
        return 'CNN'
    return 'CNN'


# ---------------------------------------------------------------------------
# Load & harmonise
# ---------------------------------------------------------------------------
def load_and_harmonise(out_dir: str) -> pd.DataFrame:
    """Load all available result CSVs and return a unified DataFrame."""
    rows = []

    # ------------------------------------------------------------------
    # 1. prod_model_summary.csv
    # ------------------------------------------------------------------
    path = os.path.join(out_dir, "prod_model_summary.csv")
    try:
        df = pd.read_csv(path)
        # Rename space-columns
        df = df.rename(columns={
            "Macro F1":    "Macro_F1",
            "Weighted F1": "Weighted_F1",
            "Train Time(s)": "Train_Time_s",
        })
        for _, r in df.iterrows():
            arch = _infer_arch(str(r['Model']))
            task = str(r.get('Task', 'multi')).strip().lower()
            rows.append({
                'Model':       str(r['Model']),
                'Task':        task,
                'Accuracy':    float(r['Accuracy']),
                'mIoU':        np.nan,
                'Macro_F1':    float(r['Macro_F1'])    if pd.notna(r.get('Macro_F1'))    else np.nan,
                'Weighted_F1': float(r['Weighted_F1']) if pd.notna(r.get('Weighted_F1')) else np.nan,
                'Architecture': arch,
                'Params_K':    PARAM_COUNTS.get(arch, 0),
                'Train_Time_s': float(r['Train_Time_s']) if pd.notna(r.get('Train_Time_s')) else np.nan,
                'Notes':       'Production baseline' if arch == 'CNN' else '',
            })
        print(f"  Loaded {len(df)} rows from prod_model_summary.csv")
    except Exception as exc:
        print(f"  WARNING: could not load prod_model_summary.csv — {exc}")

    # ------------------------------------------------------------------
    # 2. es_cnn_results.csv  (early-stopping CNN)
    # ------------------------------------------------------------------
    path = os.path.join(out_dir, "es_cnn_results.csv")
    try:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            task = str(r.get('Task', 'multi')).strip().lower()
            rows.append({
                'Model':        str(r['Model']),
                'Task':         task,
                'Accuracy':     float(r['Accuracy']),
                'mIoU':         np.nan,
                'Macro_F1':     float(r['Macro_F1'])    if pd.notna(r.get('Macro_F1'))    else np.nan,
                'Weighted_F1':  float(r['Weighted_F1']) if pd.notna(r.get('Weighted_F1')) else np.nan,
                'Architecture': 'CNN',
                'Params_K':     PARAM_COUNTS['CNN'],
                'Train_Time_s': float(r['Train_Time(s)']) if pd.notna(r.get('Train_Time(s)')) else np.nan,
                'Notes':        'With early stopping',
            })
        print(f"  Loaded {len(df)} rows from es_cnn_results.csv")
    except Exception as exc:
        print(f"  WARNING: could not load es_cnn_results.csv — {exc}")

    # ------------------------------------------------------------------
    # 3. rnn_results.csv  (FastGRU / NormalLSTM)
    # ------------------------------------------------------------------
    path = os.path.join(out_dir, "rnn_results.csv")
    try:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            raw_arch = str(r.get('Architecture', '')).strip()
            if 'gru' in raw_arch.lower() or 'fastgru' in raw_arch.lower():
                arch = 'GRU'
            elif 'lstm' in raw_arch.lower() or 'normallstm' in raw_arch.lower():
                arch = 'LSTM'
            else:
                arch = _infer_arch(str(r['Model']))
            task = str(r.get('Task', 'multi')).strip().lower()
            stopped = r.get('Stopped_Epoch', np.nan)
            rows.append({
                'Model':        str(r['Model']),
                'Task':         task,
                'Accuracy':     float(r['Accuracy']),
                'mIoU':         np.nan,
                'Macro_F1':     float(r['Macro_F1'])    if pd.notna(r.get('Macro_F1'))    else np.nan,
                'Weighted_F1':  float(r['Weighted_F1']) if pd.notna(r.get('Weighted_F1')) else np.nan,
                'Architecture': arch,
                'Params_K':     PARAM_COUNTS.get(arch, 0),
                'Train_Time_s': float(r['Train_Time(s)']) if pd.notna(r.get('Train_Time(s)')) else np.nan,
                'Notes':        f"Stopped epoch {int(stopped)}" if pd.notna(stopped) else '',
            })
        print(f"  Loaded {len(df)} rows from rnn_results.csv")
    except Exception as exc:
        print(f"  WARNING: could not load rnn_results.csv — {exc}")

    # ------------------------------------------------------------------
    # 4. unet_results.csv  (segmentation)
    # ------------------------------------------------------------------
    path = os.path.join(out_dir, "unet_results.csv")
    try:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                'Model':        str(r['Model']),
                'Task':         'segmentation',
                'Accuracy':     np.nan,
                'mIoU':         float(r['mIoU']) if pd.notna(r.get('mIoU')) else np.nan,
                'Macro_F1':     np.nan,
                'Weighted_F1':  np.nan,
                'Architecture': 'UNet',
                'Params_K':     PARAM_COUNTS['UNet'],
                'Train_Time_s': float(r['Train_Time(s)']) if pd.notna(r.get('Train_Time(s)')) else np.nan,
                'Notes':        'Segmentation (mIoU metric)',
            })
        print(f"  Loaded {len(df)} rows from unet_results.csv")
    except Exception as exc:
        print(f"  WARNING: could not load unet_results.csv — {exc}")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows, columns=[
        'Model', 'Task', 'Accuracy', 'mIoU', 'Macro_F1', 'Weighted_F1',
        'Architecture', 'Params_K', 'Train_Time_s', 'Notes',
    ])
    return result


# ---------------------------------------------------------------------------
# Plot 1 — accuracy bar chart
# ---------------------------------------------------------------------------
def plot_accuracy_comparison(df: pd.DataFrame, out_path: str) -> None:
    """Horizontal bar chart of model accuracy (multi-class + UNet mIoU)."""
    # Classification rows (multi task only)
    cls_df = df[df['Task'] == 'multi'].copy()
    cls_df['_metric'] = cls_df['Accuracy']
    cls_df['_label']  = cls_df['Accuracy'].map(lambda v: f"{v*100:.2f}%")

    # Segmentation rows (UNet) — use mIoU
    seg_df = df[df['Task'] == 'segmentation'].copy()
    seg_df['_metric'] = seg_df['mIoU']
    seg_df['_label']  = seg_df['mIoU'].map(
        lambda v: f"{v*100:.2f}% (mIoU)" if pd.notna(v) else "N/A"
    )

    plot_df = pd.concat([cls_df, seg_df], ignore_index=True)
    plot_df = plot_df.dropna(subset=['_metric'])
    plot_df = plot_df.sort_values('_metric', ascending=True)

    if plot_df.empty:
        print("  WARNING: no data to plot for accuracy comparison.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.55)))

    colours = [ARCH_COLOURS.get(a, '#888888') for a in plot_df['Architecture']]
    bars = ax.barh(plot_df['Model'], plot_df['_metric'], color=colours, edgecolor='white',
                   linewidth=0.5, height=0.65)

    # Value labels
    for bar, label in zip(bars, plot_df['_label']):
        x = bar.get_width()
        ax.text(x + 0.003, bar.get_y() + bar.get_height() / 2,
                label, va='center', ha='left', fontsize=8.5)

    # CNN baseline line
    ax.axvline(CNN_BASELINE, color='crimson', linestyle='--', linewidth=1.2,
               label=f"CNN Baseline ({CNN_BASELINE*100:.2f}%)")

    # Legend for architecture families
    seen = {}
    for arch, col in ARCH_COLOURS.items():
        if arch in plot_df['Architecture'].values:
            seen[arch] = col
    arch_handles = [
        plt.Rectangle((0, 0), 1, 1, color=col, label=arch)
        for arch, col in seen.items()
    ]
    arch_legend = ax.legend(handles=arch_handles, title='Architecture',
                            loc='lower right', fontsize=8, title_fontsize=8.5)
    ax.add_artist(arch_legend)
    ax.legend(loc='upper left', fontsize=8)

    ax.set_xlabel('Accuracy / mIoU', fontsize=10)
    ax.set_title('Model Accuracy Comparison — PPE Classification', fontsize=12, fontweight='bold')
    ax.set_xlim(0, min(plot_df['_metric'].max() * 1.15, 1.05))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved accuracy chart -> {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — per-class F1 heatmap
# ---------------------------------------------------------------------------
def plot_f1_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Per-class F1 heatmap using report_full_summary.csv data."""
    report_path = os.path.join(os.path.dirname(out_path), "report_full_summary.csv")
    try:
        rep = pd.read_csv(report_path)
    except Exception as exc:
        print(f"  WARNING: could not load report_full_summary.csv — {exc}")
        return

    f1_cols = [f"F1_{c}" for c in ALL_CLASSES]
    # Keep only multi-class rows that have at least one F1 column
    multi = rep[rep['Task'] == 'multi'].copy()
    available = [c for c in f1_cols if c in multi.columns]
    if not available:
        print("  WARNING: no per-class F1 columns found in report_full_summary.csv.")
        return

    multi = multi.dropna(subset=available, how='all')

    # If RNN results exist in the harmonised df, add stub rows (NaN F1)
    rnn_models = df[df['Architecture'].isin(['GRU', 'LSTM']) & (df['Task'] == 'multi')]
    if not rnn_models.empty:
        stub_rows = []
        for _, r in rnn_models.iterrows():
            stub = {'Model': r['Model'], 'Task': 'multi'}
            for c in available:
                stub[c] = np.nan
            stub_rows.append(stub)
        stub_df = pd.DataFrame(stub_rows)
        multi = pd.concat([multi, stub_df], ignore_index=True)

    heatmap_data = multi.set_index('Model')[available]
    heatmap_data.columns = [c.replace('F1_', '').replace('_', ' ') for c in available]
    heatmap_data = heatmap_data.astype(float)

    fig, ax = plt.subplots(figsize=(max(7, len(available) * 1.4), max(4, len(heatmap_data) * 0.7)))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'F1 Score', 'shrink': 0.8},
        annot_kws={'size': 9},
    )
    ax.set_title('Per-Class F1 Score Heatmap', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('PPE Class', fontsize=10)
    ax.set_ylabel('Model', fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved F1 heatmap -> {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def make_latex_table(df: pd.DataFrame, out_path: str) -> None:
    """Write a LaTeX tabular table of multi-class + segmentation results."""
    sub = df[df['Task'].isin(['multi', 'segmentation'])].copy()
    if sub.empty:
        print("  WARNING: no rows for LaTeX table.")
        return

    # Identify best values for bolding
    best_acc  = sub['Accuracy'].max()   if sub['Accuracy'].notna().any()  else None
    best_miou = sub['mIoU'].max()       if sub['mIoU'].notna().any()      else None

    def _fmt_metric(val, best) -> str:
        if pd.isna(val):
            return '—'
        s = f"{val:.4f}"
        return rf'\textbf{{{s}}}' if (best is not None and abs(val - best) < 1e-9) else s

    def _fmt_int(val) -> str:
        if pd.isna(val) or val == 0:
            return '0'
        return str(int(val))

    col_spec = 'llrrrr'
    header   = (r'Model & Architecture & Accuracy & mIoU & Macro F1 & Params (K) \\ \midrule')

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{PPE Detection Model Comparison}',
        r'\label{tab:model_comparison}',
        rf'\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        header,
    ]

    for _, r in sub.iterrows():
        acc_str  = _fmt_metric(r['Accuracy'], best_acc)
        miou_str = _fmt_metric(r['mIoU'],     best_miou)
        f1_str   = f"{r['Macro_F1']:.4f}" if pd.notna(r.get('Macro_F1')) else '—'
        pk_str   = _fmt_int(r.get('Params_K', np.nan))

        # Escape underscores in model names for LaTeX
        model_name = str(r['Model']).replace('_', r'\_').replace('%', r'\%')
        arch_name  = str(r['Architecture']).replace('_', r'\_')

        line = f"{model_name} & {arch_name} & {acc_str} & {miou_str} & {f1_str} & {pk_str} \\\\"
        lines.append(line)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  Saved LaTeX table -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("PPE Experiment Comparison")
    print("=" * 60)
    print(f"\nScanning: {OUT_DIR}\n")

    df = load_and_harmonise(OUT_DIR)

    if df.empty:
        print("\nNo result CSVs found. Run the experiment scripts first.")
        sys.exit(0)

    print(f"\nLoaded {len(df)} model results")
    print(df[['Model', 'Task', 'Accuracy', 'mIoU', 'Macro_F1', 'Architecture', 'Params_K']].to_string(index=False))

    plot_accuracy_comparison(df, os.path.join(OUT_DIR, "experiment_comparison.png"))
    plot_f1_heatmap(df, os.path.join(OUT_DIR, "experiment_f1_heatmap.png"))
    make_latex_table(df, os.path.join(OUT_DIR, "experiment_table.tex"))

    df.to_csv(os.path.join(OUT_DIR, "experiment_comparison_full.csv"), index=False)
    print(f"\nOutputs saved to {OUT_DIR}")
