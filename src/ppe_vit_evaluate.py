"""
Evaluate trained ViT model and write prod_vit_results.csv
for ingestion by ppe_experiment_comparison.py
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Project root = two levels up from src/ (works for both main tree and worktrees)
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT_DIR     = os.path.join(PROJECT_DIR, "results", "models")

    # Dataset path: check known locations in priority order
    for _candidate in [
        "D:/datasets/ppe_crops",
        "D:/Claude/datasets/ppe_crops",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "datasets", "ppe_crops"),
    ]:
        if os.path.exists(_candidate):
            DATA_DIR = _candidate
            break
    else:
        raise FileNotFoundError("ppe_crops dataset not found. Run setup_datasets.py first.")
    CKPT = os.path.join(OUT_DIR, "prod_vit_model.pth")
    DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("PPE ViT EVALUATION")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_tf)
    val_ldr = DataLoader(val_ds, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)
    classes = val_ds.classes
    print(f"Val samples: {len(val_ds)}  Classes: {classes}")

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(classes))

    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    # Checkpoint may use 'state_dict' or 'model_state_dict' as key
    if isinstance(ckpt, dict):
        state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f"Loaded checkpoint: {CKPT}")

    all_pred, all_true = [], []
    t0 = time.time()
    with torch.no_grad():
        for xb, yb in val_ldr:
            preds = model(xb.to(DEVICE)).argmax(1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(yb.tolist())
    elapsed = time.time() - t0

    rep = classification_report(
        all_true, all_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    acc         = rep['accuracy']
    macro_f1    = rep['macro avg']['f1-score']
    weighted_f1 = rep['weighted avg']['f1-score']

    print(f"\nVal Acc:     {acc:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Eval time:   {elapsed:.1f}s")
    print()
    print(classification_report(all_true, all_pred, target_names=classes, zero_division=0))

    csv_path = os.path.join(OUT_DIR, "prod_vit_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'Model', 'Task', 'Accuracy', 'Macro_F1', 'Weighted_F1',
            'Train_Time(s)', 'Architecture', 'Params_K', 'Notes',
        ])
        w.writerow([
            'ViT-B-16 (fine-tuned)', 'multi',
            f'{acc:.4f}', f'{macro_f1:.4f}', f'{weighted_f1:.4f}',
            '',
            'ViT', '86000',
            'ImageNet pretrained, 20ep fine-tune',
        ])
    print(f"Saved {csv_path}")

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'ViT-B-16 Confusion Matrix  (val acc={acc:.3f})')
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, "prod_vit_confusion.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved {cm_path}")


if __name__ == '__main__':
    main()
