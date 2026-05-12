"""
Masked CNN: PPENetFast trained on SAM2-masked crops.

For MinhNKB images that have a SAM2 pseudo-mask, the mask is applied to the
source image before person crops are extracted.  For Jomarkow crops (no masks),
an elliptical centre-foreground mask is used instead.

Both strategies aim to zero out scene background so the classifier focuses on
the person's PPE rather than scene clutter.
"""
import os
import time
import csv
import random
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
for _cand in ["D:/datasets", "D:/Claude/datasets",
              os.path.join(os.path.dirname(os.path.dirname(
                  os.path.dirname(os.path.abspath(__file__)))), "datasets")]:
    if os.path.exists(os.path.join(_cand, "jomarkow")):
        DATASETS = _cand
        break
else:
    raise FileNotFoundError("datasets/ not found")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR     = os.path.join(PROJECT_DIR, "results", "models")
MASK_DIR    = os.path.join(DATASETS, "ppe_masks")

MINHNKB_IMG = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-data")
MINHNKB_ANN = os.path.join(DATASETS, "helmet-safety-vest-detection-master/train-images-annotations-new")
JOMARK_IMG  = os.path.join(DATASETS, "jomarkow/images")
JOMARK_LBL  = os.path.join(DATASETS, "jomarkow/labels")

CLASSES    = ["helmet", "safety_vest", "full_ppe", "partial_ppe", "no_ppe"]
CROP_SIZE  = 32
MAX_CLASS  = 600
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Annotation maps ────────────────────────────────────────────────────────
MINHNKB_MAP = {
    "helmet": "helmet", "safety vest": "safety_vest",
    "person with full safety": "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety": "no_ppe",
    "head": "no_ppe", "person": "no_ppe", "safety_vest": "safety_vest",
    "vest": "safety_vest", "no_helmet": "no_ppe", "helmet_vest": "full_ppe",
    "helmet_novest": "partial_ppe", "nohelmet_vest": "partial_ppe",
    "nohelmet_novest": "no_ppe",
}
JOMARK_MAP  = {0: "helmet", 1: "no_ppe"}


# ── Helpers ────────────────────────────────────────────────────────────────
def load_mask(stem: str):
    """Return binary mask (H×W uint8) for a MinhNKB image stem, or None."""
    path = os.path.join(MASK_DIR, f"{stem}_mask.png")
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8) if m is not None else None


def apply_mask(img_bgr, mask):
    """Zero out background pixels given a binary foreground mask."""
    if mask is None or img_bgr is None:
        return img_bgr
    if mask.shape != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    out = img_bgr.copy()
    out[mask == 0] = 0
    return out


def ellipse_mask(h, w):
    """Elliptical centre-foreground mask (fallback for non-masked datasets)."""
    m = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    ry, rx = max(1, int(h * 0.45)), max(1, int(w * 0.45))
    cv2.ellipse(m, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    return m


def extract_crop(img_bgr, x1, y1, x2, y2, use_ellipse=False):
    """Crop a bounding box and resize to CROP_SIZE×CROP_SIZE."""
    h_img, w_img = img_bgr.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_img, x2); y2 = min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    if use_ellipse:
        em = ellipse_mask(*crop.shape[:2])
        crop = apply_mask(crop, em)
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    return crop


# ── Dataset builders ───────────────────────────────────────────────────────
def parse_minhnkb(crops_by_cls):
    ann_files = sorted(os.listdir(MINHNKB_ANN))
    for ann_file in ann_files:
        if not ann_file.endswith('.xml'):
            continue
        try:
            tree = ET.parse(os.path.join(MINHNKB_ANN, ann_file))
        except ET.ParseError:
            continue
        root = tree.getroot()
        fname = root.findtext('filename', '').strip()
        if not fname:
            stem = os.path.splitext(ann_file)[0]
            # find image with matching stem
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = os.path.join(MINHNKB_IMG, stem + ext)
                if os.path.exists(candidate):
                    fname = stem + ext; break
        img_path = os.path.join(MINHNKB_IMG, fname)
        if not os.path.exists(img_path):
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask = load_mask(stem)
        if mask is not None:
            img_bgr = apply_mask(img_bgr, mask)

        for obj in root.findall('object'):
            label_raw = obj.findtext('name', '').strip().lower()
            label = MINHNKB_MAP.get(label_raw)
            if label not in CLASSES:
                continue
            bb = obj.find('bndbox')
            if bb is None:
                continue
            try:
                x1, y1 = int(float(bb.findtext('xmin'))), int(float(bb.findtext('ymin')))
                x2, y2 = int(float(bb.findtext('xmax'))), int(float(bb.findtext('ymax')))
            except (TypeError, ValueError):
                continue
            crop = extract_crop(img_bgr, x1, y1, x2, y2, use_ellipse=False)
            if crop is not None:
                crops_by_cls[label].append(crop)


def parse_jomarkow(crops_by_cls):
    for img_file in sorted(os.listdir(JOMARK_IMG)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = os.path.splitext(img_file)[0]
        lbl_path = os.path.join(JOMARK_LBL, stem + '.txt')
        if not os.path.exists(lbl_path):
            continue
        img_bgr = cv2.imread(os.path.join(JOMARK_IMG, img_file))
        if img_bgr is None:
            continue
        h_img, w_img = img_bgr.shape[:2]
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                label = JOMARK_MAP.get(cls_id)
                if label not in CLASSES:
                    continue
                cx_n, cy_n, bw_n, bh_n = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx_n - bw_n / 2) * w_img)
                y1 = int((cy_n - bh_n / 2) * h_img)
                x2 = int((cx_n + bw_n / 2) * w_img)
                y2 = int((cy_n + bh_n / 2) * h_img)
                crop = extract_crop(img_bgr, x1, y1, x2, y2, use_ellipse=True)
                if crop is not None:
                    crops_by_cls[label].append(crop)


# ── CNN ────────────────────────────────────────────────────────────────────
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
        self.features   = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.pool       = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 256),     nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS, BATCH, LR = 60, 256, 1e-3
    print("=" * 65)
    print("PPE MASKED CNN TRAINING  (SAM2 masks + ellipse fallback)")
    print(f"Device: {DEVICE}  |  Crop: {CROP_SIZE}×{CROP_SIZE}  |  Epochs: {EPOCHS}")
    print("=" * 65)

    # Build masked crops
    crops_by_cls = {c: [] for c in CLASSES}
    print("\nParsing MinhNKB (SAM2-masked)…")
    parse_minhnkb(crops_by_cls)
    print("Parsing Jomarkow (ellipse-masked)…")
    parse_jomarkow(crops_by_cls)

    for c in CLASSES:
        print(f"  {c:15s}: {len(crops_by_cls[c])} crops")

    # Balance
    X, y = [], []
    for cls, crops in crops_by_cls.items():
        random.shuffle(crops)
        for crop in crops[:MAX_CLASS]:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            X.append(rgb.transpose(2, 0, 1))
            y.append(cls)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_arr = np.array(X, dtype=np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X_arr, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Classes: {list(le.classes_)}")

    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std  = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    train_ds  = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    val_ds    = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val,   dtype=torch.long))
    train_ldr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, pin_memory=True)

    model     = PPENetFast(num_classes=len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_ldr))

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc, best_state = 0.0, None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for xb, yb in train_ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward(); optimizer.step(); scheduler.step()
            tl += loss.item() * len(xb)
            tc += (out.argmax(1) == yb).sum().item()
            tt += len(xb)

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_ldr:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                vl += criterion(out, yb).item() * len(xb)
                preds = out.argmax(1)
                vc += (preds == yb).sum().item()
                vt += len(xb)
                all_pred.extend(preds.cpu().tolist())
                all_true.extend(yb.cpu().tolist())

        ta, va = tc / tt, vc / vt
        history['train_loss'].append(tl / tt)
        history['val_loss'].append(vl / vt)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)
        if va > best_acc:
            best_acc  = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"Ep {ep:3d}/{EPOCHS}  train={ta:.4f}  val={va:.4f}  best={best_acc:.4f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f}m  best_val_acc={best_acc:.4f}")

    model.load_state_dict(best_state)
    torch.save({'model_state_dict': best_state, 'classes': list(le.classes_),
                'best_val_acc': best_acc},
               os.path.join(OUT_DIR, "masked_cnn_model.pth"))

    # Final evaluation
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_ldr:
            all_pred.extend(model(xb.to(DEVICE)).argmax(1).cpu().tolist())
            all_true.extend(yb.tolist())

    rep        = classification_report(all_true, all_pred,
                                       target_names=list(le.classes_),
                                       output_dict=True, zero_division=0)
    acc        = rep['accuracy']
    macro_f1   = rep['macro avg']['f1-score']
    wf1        = rep['weighted avg']['f1-score']
    print(classification_report(all_true, all_pred, target_names=list(le.classes_), zero_division=0))

    with open(os.path.join(OUT_DIR, "masked_cnn_results.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'Task', 'Accuracy', 'Macro_F1', 'Weighted_F1', 'Train_Time(s)'])
        w.writerow(['MaskedCNN (SAM2+ellipse)', 'multi',
                    f'{acc:.4f}', f'{macro_f1:.4f}', f'{wf1:.4f}', f'{elapsed:.1f}'])

    eps = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(eps, history['train_loss'], 'r-', label='Train')
    ax1.plot(eps, history['val_loss'],   'b-', label='Val')
    ax1.set_title('Masked-CNN Loss'); ax1.legend()
    ax2.plot(eps, history['train_acc'], 'r-', label='Train')
    ax2.plot(eps, history['val_acc'],   'b-', label='Val')
    ax2.axhline(best_acc, color='g', linestyle='--', label=f'Best: {best_acc:.3f}')
    ax2.set_title('Masked-CNN Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "masked_cnn_training.png"), dpi=150)
    plt.close()

    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(le.classes_), yticklabels=list(le.classes_), ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Masked-CNN Confusion Matrix  (acc={acc:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "masked_cnn_confusion.png"), dpi=150)
    plt.close()

    print(f"\nSaved: masked_cnn_model.pth  masked_cnn_results.csv  masked_cnn_training.png  masked_cnn_confusion.png")
    print(f"Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}  vs baseline 87.33%")


if __name__ == '__main__':
    main()
