# Finish PPE Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the four outstanding gaps in the PPE project: fix the missing ViT result in experiment comparison, train YOLO end-to-end PPE detection, build a synthetic-masked CNN classifier, and regenerate the final report — pushing to main after each task.

**Architecture:** Each task is self-contained. ViT fix adds a CSV write + new loader. YOLO end-to-end uses the existing setup_yolo_ppe.py then standard `yolo train`. Masked-CNN re-uses the SAM2 mask index to zero-out image backgrounds before feeding the existing PPENetFast architecture. Final report regeneration runs the existing generate_assignment_report.py script.

**Tech Stack:** PyTorch, torchvision, ultralytics (YOLO), sklearn, python-docx, matplotlib, seaborn, pandas, numpy

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `src/ppe_vit_evaluate.py` | Create | Load prod_vit_model.pth, eval on test set, save prod_vit_results.csv |
| `src/ppe_experiment_comparison.py` | Modify | Add loader for prod_vit_results.csv (Task 1 only) |
| `src/ppe_masked_cnn_train.py` | Create | Train PPENetFast on SAM2-masked crops, save masked_cnn_results.csv |
| `results/models/prod_vit_results.csv` | Create | ViT accuracy/F1 metrics for comparison ingestion |
| `results/models/masked_cnn_results.csv` | Create | Masked-CNN accuracy/F1 for comparison ingestion |
| `results/models/masked_cnn_model.pth` | Create | Saved masked-CNN weights |
| `results/models/masked_cnn_confusion.png` | Create | Confusion matrix for masked CNN |
| `results/models/masked_cnn_training.png` | Create | Training curves for masked CNN |
| `docs/Final_Project_Writeup.docx` | Overwrite | Regenerated report with all models |

---

### Task 1: Fix ViT in Experiment Comparison

**Files:**
- Create: `src/ppe_vit_evaluate.py`
- Modify: `src/ppe_experiment_comparison.py`
- Output: `results/models/prod_vit_results.csv`

- [ ] **Step 1: Create ppe_vit_evaluate.py** — load checkpoint, run test set eval, write CSV

```python
"""
Evaluate trained ViT model and write prod_vit_results.csv
for ingestion by ppe_experiment_comparison.py
"""
import os, time, csv
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ViT_B_16_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = "D:/datasets/ppe_crops" if os.path.exists("D:/datasets/ppe_crops") else os.path.join(BASE, "datasets", "ppe_crops")
    OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models")
    CKPT     = os.path.join(OUT_DIR, "prod_vit_model.pth")
    DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), val_tf)
    val_ldr = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    classes = val_ds.classes

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(classes))
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    all_pred, all_true = [], []
    t0 = time.time()
    with torch.no_grad():
        for xb, yb in val_ldr:
            all_pred.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
            all_true.extend(yb.tolist())
    elapsed = time.time() - t0

    rep = classification_report(all_true, all_pred, target_names=classes, output_dict=True, zero_division=0)
    acc = rep['accuracy']
    macro_f1  = rep['macro avg']['f1-score']
    weighted_f1 = rep['weighted avg']['f1-score']

    print(f"ViT Val Acc: {acc:.4f}  Macro-F1: {macro_f1:.4f}  Time: {elapsed:.1f}s")

    with open(os.path.join(OUT_DIR, "prod_vit_results.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'Task', 'Accuracy', 'Macro_F1', 'Weighted_F1', 'Train_Time(s)', 'Architecture', 'Params_K', 'Notes'])
        w.writerow(['ViT-B-16 (fine-tuned)', 'multi', f'{acc:.4f}', f'{macro_f1:.4f}', f'{weighted_f1:.4f}', '', 'ViT', '86000', 'ImageNet pretrained, 20ep fine-tune'])

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'ViT-B-16 Confusion Matrix (val acc={acc:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prod_vit_confusion.png"), dpi=150)
    plt.close()
    print("Saved prod_vit_results.csv and prod_vit_confusion.png")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run evaluation**

```bash
python src/ppe_vit_evaluate.py
```

Expected: prints ViT Val Acc ~0.939, creates `results/models/prod_vit_results.csv`

- [ ] **Step 3: Add ViT loader to ppe_experiment_comparison.py**

After the `unet_results.csv` block (around line 215), add:

```python
    # ------------------------------------------------------------------
    # 5. prod_vit_results.csv  (ViT fine-tune)
    # ------------------------------------------------------------------
    path = os.path.join(out_dir, "prod_vit_results.csv")
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
                'Architecture': 'ViT',
                'Params_K':     86000,
                'Train_Time_s': float(r['Train_Time(s)']) if pd.notna(r.get('Train_Time(s)')) else np.nan,
                'Notes':        str(r.get('Notes', '')),
            })
        print(f"  Loaded {len(df)} rows from prod_vit_results.csv")
    except Exception as exc:
        print(f"  WARNING: could not load prod_vit_results.csv -- {exc}")
```

Also add `'ViT': '#E07B54'` to `ARCH_COLOURS` and `'ViT': 86000` to `PARAM_COUNTS`.

- [ ] **Step 4: Re-run experiment comparison**

```bash
python src/ppe_experiment_comparison.py
```

Expected: prints 15+ model results; ViT appears at top of accuracy chart

- [ ] **Step 5: Commit and push to main**

```bash
git add src/ppe_vit_evaluate.py src/ppe_experiment_comparison.py results/models/prod_vit_results.csv results/models/experiment_comparison.png results/models/experiment_comparison_full.csv results/models/experiment_f1_heatmap.png results/models/experiment_table.tex results/models/prod_vit_confusion.png
git commit -m "feat: add ViT evaluation script; fix ViT missing from experiment comparison"
git checkout main && git merge --ff-only claude/flamboyant-payne-c40794 && git push origin main && git checkout claude/flamboyant-payne-c40794
```

---

### Task 2: YOLO End-to-End PPE Detection

**Files:**
- Run: `src/setup_yolo_ppe.py`
- Creates: `D:/datasets/yolo_ppe_end2end/`
- Output: `runs/detect/yolov8_ppe_e2e_prod/` (YOLO standard output)

- [ ] **Step 1: Run dataset setup**

```bash
python src/setup_yolo_ppe.py
```

Expected: creates `D:/datasets/yolo_ppe_end2end/{train,val}/{images,labels}/` with YAML config

- [ ] **Step 2: Train YOLO end-to-end**

```bash
yolo train data=D:/datasets/yolo_ppe_end2end/yolo_ppe.yaml model=yolov8n.pt epochs=30 imgsz=416 batch=32 device=0 name=yolov8_ppe_e2e_prod project=runs/detect
```

Expected: trains ~30 epochs, saves best.pt to `runs/detect/yolov8_ppe_e2e_prod/weights/best.pt`

- [ ] **Step 3: Validate**

```bash
yolo val data=D:/datasets/yolo_ppe_end2end/yolo_ppe.yaml model=runs/detect/yolov8_ppe_e2e_prod/weights/best.pt device=0
```

Expected: prints mAP50, precision, recall per class

- [ ] **Step 4: Commit and push to main**

```bash
git add runs/detect/yolov8_ppe_e2e_prod/confusion_matrix.png runs/detect/yolov8_ppe_e2e_prod/results.png
git commit -m "results: YOLO end-to-end PPE detection training"
git checkout main && git merge --ff-only claude/flamboyant-payne-c40794 && git push origin main && git checkout claude/flamboyant-payne-c40794
```

---

### Task 3: Synthetic-Masked CNN Model

Uses existing SAM2 pseudo-masks (mask_index.csv, 2609 entries) to zero out image backgrounds before training a PPENetFast classifier. Hypothesis: removing background noise should improve accuracy over the unmasked 87.33% baseline.

**Files:**
- Create: `src/ppe_masked_cnn_train.py`
- Output: `results/models/masked_cnn_model.pth`, `masked_cnn_results.csv`, `masked_cnn_training.png`, `masked_cnn_confusion.png`
- Modify: `src/ppe_experiment_comparison.py` — add loader for `masked_cnn_results.csv`

- [ ] **Step 1: Create ppe_masked_cnn_train.py**

```python
"""
Masked CNN: train PPENetFast on SAM2-masked crops.
Loads mask_index.csv, applies binary masks to remove backgrounds,
then trains with same hyperparams as production CNN.
Falls back to unmasked training if mask is unavailable for a sample.
"""
import os, time, csv, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# ---- PPENetFast (same as production) -----------------------------------
class PPENetFast(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3,32), block(32,64), block(64,128))
        self.pool = nn.AdaptiveAvgPool2d((2,2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 256),   nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

# ---- Masked dataset -----------------------------------------------------
class MaskedCropDataset(Dataset):
    def __init__(self, crop_dir, mask_lookup, split, transform):
        self.transform = transform
        self.samples = []
        classes = sorted(os.listdir(os.path.join(crop_dir, split)))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            folder = os.path.join(crop_dir, split, cls)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder, fname)
                    mask_path = mask_lookup.get(img_path)
                    self.samples.append((img_path, mask_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L').resize(img.size, Image.NEAREST)
            mask_arr = np.array(mask) > 127
            img_arr  = np.array(img)
            img_arr[~mask_arr] = 0  # zero out background
            img = Image.fromarray(img_arr)
        return self.transform(img), label

def build_mask_lookup(mask_index_path):
    """Return dict: image_path -> mask_path"""
    if not os.path.exists(mask_index_path):
        return {}
    df = pd.read_csv(mask_index_path)
    return dict(zip(df['image_path'], df['mask_path']))

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    BASE     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = "D:/datasets/ppe_crops" if os.path.exists("D:/datasets/ppe_crops") else os.path.join(BASE, "datasets", "ppe_crops")
    OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models")
    MASK_IDX = os.path.join(OUT_DIR, "mask_index.csv")
    DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS, BATCH, LR = 60, 256, 1e-3

    mask_lookup = build_mask_lookup(MASK_IDX)
    print(f"Mask lookup: {len(mask_lookup)} entries")

    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = MaskedCropDataset(DATA_DIR, mask_lookup, 'train', train_tf)
    val_ds   = MaskedCropDataset(DATA_DIR, mask_lookup, 'val',   val_tf)
    classes  = list(train_ds.class_to_idx.keys())

    train_ldr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = PPENetFast(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_ldr))

    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_acc, best_state = 0.0, None
    t0 = time.time()

    for ep in range(1, EPOCHS+1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for xb, yb in train_ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step(); scheduler.step()
            tl += loss.item()*len(xb); tc += (model(xb).argmax(1)==yb).sum().item(); tt += len(xb)
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_ldr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                vl += criterion(out, yb).item()*len(xb)
                preds = out.argmax(1)
                vc += (preds==yb).sum().item(); vt += len(xb)
                all_pred.extend(preds.cpu().tolist())
                all_true.extend(yb.cpu().tolist())
        ta, va = tc/tt, vc/vt
        history['train_loss'].append(tl/tt); history['val_loss'].append(vl/vt)
        history['train_acc'].append(ta);     history['val_acc'].append(va)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"Ep {ep:3d}/{EPOCHS}  train={ta:.4f}  val={va:.4f}  best={best_acc:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m  best_val_acc={best_acc:.4f}")

    model.load_state_dict(best_state)
    torch.save({'model_state_dict': best_state, 'classes': classes, 'best_val_acc': best_acc},
               os.path.join(OUT_DIR, "masked_cnn_model.pth"))

    # Final evaluation
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_ldr:
            all_pred.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
            all_true.extend(yb.tolist())
    rep = classification_report(all_true, all_pred, target_names=classes, output_dict=True, zero_division=0)
    acc = rep['accuracy']; macro_f1 = rep['macro avg']['f1-score']; wf1 = rep['weighted avg']['f1-score']

    with open(os.path.join(OUT_DIR, "masked_cnn_results.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model','Task','Accuracy','Macro_F1','Weighted_F1','Train_Time(s)'])
        w.writerow(['MaskedCNN (SAM2 bg-removed)', 'multi', f'{acc:.4f}', f'{macro_f1:.4f}', f'{wf1:.4f}', f'{elapsed:.1f}'])

    # Training curves
    eps = range(1, EPOCHS+1)
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(eps, history['train_loss'], 'r-', label='Train'); ax1.plot(eps, history['val_loss'], 'b-', label='Val')
    ax1.set_title('Masked-CNN Loss'); ax1.legend()
    ax2.plot(eps, history['train_acc'], 'r-', label='Train'); ax2.plot(eps, history['val_acc'], 'b-', label='Val')
    ax2.axhline(best_acc, color='g', linestyle='--', label=f'Best: {best_acc:.3f}')
    ax2.set_title('Masked-CNN Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "masked_cnn_training.png"), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Masked-CNN Confusion Matrix (acc={acc:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "masked_cnn_confusion.png"), dpi=150)
    plt.close()
    print(f"Results saved: acc={acc:.4f}  macro_f1={macro_f1:.4f}")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run masked CNN training**

```bash
python src/ppe_masked_cnn_train.py
```

Expected: trains 60 epochs, ~3-5 min on RTX 5070; saves masked_cnn_results.csv with accuracy vs 87.33% baseline

- [ ] **Step 3: Add masked_cnn loader to ppe_experiment_comparison.py**

After the ViT loader block, add:

```python
    # ------------------------------------------------------------------
    # 6. masked_cnn_results.csv  (SAM2-masked background removal CNN)
    # ------------------------------------------------------------------
    path = os.path.join(out_dir, "masked_cnn_results.csv")
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
                'Params_K':     226,
                'Train_Time_s': float(r['Train_Time(s)']) if pd.notna(r.get('Train_Time(s)')) else np.nan,
                'Notes':        'SAM2 background removal',
            })
        print(f"  Loaded {len(df)} rows from masked_cnn_results.csv")
    except Exception as exc:
        print(f"  WARNING: could not load masked_cnn_results.csv -- {exc}")
```

- [ ] **Step 4: Re-run experiment comparison**

```bash
python src/ppe_experiment_comparison.py
```

Expected: 16+ models in output; MaskedCNN appears in chart labelled "MaskedCNN (SAM2 bg-removed)"

- [ ] **Step 5: Commit and push to main**

```bash
git add src/ppe_masked_cnn_train.py src/ppe_experiment_comparison.py results/models/masked_cnn_model.pth results/models/masked_cnn_results.csv results/models/masked_cnn_training.png results/models/masked_cnn_confusion.png results/models/experiment_comparison.png results/models/experiment_comparison_full.csv results/models/experiment_f1_heatmap.png results/models/experiment_table.tex
git commit -m "feat: add SAM2-masked CNN; update experiment comparison with ViT + masked-CNN"
git checkout main && git merge --ff-only claude/flamboyant-payne-c40794 && git push origin main && git checkout claude/flamboyant-payne-c40794
```

---

### Task 4: Regenerate Final Report

**Files:**
- Modify: `reports/generate_assignment_report.py` — add masked-CNN and ViT rows to table, add confusion matrix images
- Overwrite: `docs/Final_Project_Writeup.docx`

- [ ] **Step 1: Run report generation**

```bash
cd D:\Claude\PPE-Detection\.claude\worktrees\flamboyant-payne-c40794 && python reports/generate_assignment_report.py
```

Expected: saves `docs/Final_Project_Writeup.docx`

- [ ] **Step 2: Commit and push to main**

```bash
git add docs/Final_Project_Writeup.docx reports/generate_assignment_report.py
git commit -m "docs: regenerate final writeup with ViT + masked-CNN + YOLO results"
git checkout main && git merge --ff-only claude/flamboyant-payne-c40794 && git push origin main && git checkout claude/flamboyant-payne-c40794
```
