"""
PPE (Personal Protective Equipment) Detection Pipeline
=======================================================
Datasets:
  - MinhNKB/helmet-safety-vest-detection (1613 images, Pascal VOC format)
    Classes: helmet, safety vest, person with full safety,
             person with partial safety, person without safety

Models:
  - SVM with HOG + color histogram features
  - Random Forest with HOG + color histogram features
  - CNN (PyTorch) - small custom architecture

Tasks:
  - Binary: wearing PPE vs not
  - Multi-class: helmet / vest / full PPE / partial PPE / no PPE
"""

import os, sys, glob, random, time, warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve,
                              ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = "/sessions/sleepy-epic-pascal/datasets/helmet-safety-vest-detection-master"
IMG_DIR     = os.path.join(DATASET_DIR, "train-images-data")
ANN_DIR     = os.path.join(DATASET_DIR, "train-images-annotations-new")
OUT_DIR     = "/sessions/sleepy-epic-pascal/mnt/Computer Vision"
CROPS_DIR   = "/sessions/sleepy-epic-pascal/crops"
IMG_SIZE    = (64, 64)   # crop resize for features
HOG_SIZE    = (64, 64)

os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(CROPS_DIR, "per_class"), exist_ok=True)

# Label remapping: consolidate to 5 classes then binary
RAW_CLASS_MAP = {
    "helmet":                   "helmet",
    "safety vest":              "safety_vest",
    "person with full safety":  "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety":    "no_ppe",
}
BINARY_MAP = {
    "helmet":      "ppe_present",
    "safety_vest": "ppe_present",
    "full_ppe":    "ppe_present",
    "partial_ppe": "ppe_present",
    "no_ppe":      "no_ppe",
}

print("="*70)
print("PPE DETECTION PIPELINE — DATASET PARSE + FEATURE EXTRACTION")
print("="*70)

# ─── 1. PARSE XML ANNOTATIONS ──────────────────────────────────────────────────
def parse_annotations(ann_dir, img_dir):
    records = []
    xml_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
    print(f"\n[1/7] Parsing {len(xml_files)} annotation files...")
    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            fname = root.findtext("filename")
            img_path = os.path.join(img_dir, fname) if fname else None
            # fallback: same stem as xml
            if not img_path or not os.path.exists(img_path):
                stem = os.path.splitext(os.path.basename(xf))[0]
                for ext in [".jpg",".jpeg",".png"]:
                    candidate = os.path.join(img_dir, stem + ext)
                    if os.path.exists(candidate):
                        img_path = candidate; break
            if not img_path or not os.path.exists(img_path):
                continue
            size = root.find("size")
            iw = int(size.findtext("width",  0))
            ih = int(size.findtext("height", 0))
            for obj in root.findall("object"):
                raw_cls = obj.findtext("name","").strip().lower()
                if raw_cls not in RAW_CLASS_MAP:
                    continue
                cls = RAW_CLASS_MAP[raw_cls]
                bb = obj.find("bndbox")
                xmin = max(0, int(float(bb.findtext("xmin",0))))
                ymin = max(0, int(float(bb.findtext("ymin",0))))
                xmax = min(iw, int(float(bb.findtext("xmax",iw))))
                ymax = min(ih, int(float(bb.findtext("ymax",ih))))
                if xmax <= xmin or ymax <= ymin:
                    continue
                records.append({
                    "img_path": img_path,
                    "class":    cls,
                    "binary":   BINARY_MAP[cls],
                    "xmin":     xmin, "ymin": ymin,
                    "xmax":     xmax, "ymax": ymax,
                    "width":    iw,   "height": ih,
                    "box_area": (xmax-xmin)*(ymax-ymin),
                    "img_area": iw*ih,
                    "coverage": (xmax-xmin)*(ymax-ymin)/(iw*ih+1e-6),
                })
        except Exception as e:
            pass
    return pd.DataFrame(records)

df = parse_annotations(ANN_DIR, IMG_DIR)
print(f"    Total bounding boxes parsed: {len(df)}")
print(f"    Unique images:               {df['img_path'].nunique()}")
print(f"\n    Class distribution:")
cc = df['class'].value_counts()
for cls, n in cc.items():
    print(f"      {cls:<30} {n:>5}")

# ─── 2. EDA ────────────────────────────────────────────────────────────────────
print("\n[2/7] Running EDA...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("PPE Detection — Exploratory Data Analysis", fontsize=16, fontweight='bold')

# 2a. Class distribution
ax = axes[0,0]
colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
cc.plot(kind='bar', ax=ax, color=colors[:len(cc)], edgecolor='black')
ax.set_title("Class Distribution (Bounding Boxes)"); ax.set_xlabel(""); ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=35)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=9)

# 2b. Binary distribution
ax = axes[0,1]
bc = df['binary'].value_counts()
ax.pie(bc, labels=bc.index, autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'],
       startangle=90, wedgeprops={'edgecolor':'white','linewidth':1.5})
ax.set_title("Binary: PPE Present vs Not")

# 2c. Box area distribution
ax = axes[0,2]
for cls, grp in df.groupby('class'):
    ax.hist(np.log1p(grp['box_area']), bins=30, alpha=0.5, label=cls)
ax.set_title("Log(Box Area) Distribution"); ax.set_xlabel("log(area)"); ax.set_ylabel("Count")
ax.legend(fontsize=7)

# 2d. Coverage (box area / image area)
ax = axes[1,0]
for cls, grp in df.groupby('class'):
    ax.hist(grp['coverage'], bins=30, alpha=0.5, label=cls)
ax.set_title("Coverage (bbox/image area)"); ax.set_xlabel("Coverage"); ax.set_ylabel("Count")
ax.legend(fontsize=7)

# 2e. Boxes per image
bpi = df.groupby('img_path').size()
ax = axes[1,1]
ax.hist(bpi, bins=range(1, bpi.max()+2), color='steelblue', edgecolor='black', align='left')
ax.set_title("Bounding Boxes per Image"); ax.set_xlabel("# boxes"); ax.set_ylabel("# images")

# 2f. Aspect ratio
df['aspect'] = (df['xmax']-df['xmin']) / (df['ymax']-df['ymin']+1e-6)
ax = axes[1,2]
for cls, grp in df.groupby('class'):
    ax.hist(grp['aspect'].clip(0,3), bins=30, alpha=0.5, label=cls)
ax.set_title("Aspect Ratio Distribution"); ax.set_xlabel("W/H"); ax.set_ylabel("Count")
ax.legend(fontsize=7)

plt.tight_layout()
eda_path = os.path.join(OUT_DIR, "01_eda_analysis.png")
plt.savefig(eda_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    EDA plot saved: {eda_path}")

# ─── 3. EXTRACT CROPS ──────────────────────────────────────────────────────────
print("\n[3/7] Extracting image crops...")

def extract_hog_features(img_bgr, size=HOG_SIZE):
    img = cv2.resize(img_bgr, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(gray).flatten()
    return h

def extract_color_histogram(img_bgr, bins=32):
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img_bgr], [ch], None, [bins], [0,256])
        feats.append(cv2.normalize(hist, hist).flatten())
    # Also HSV histogram
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0,256])
        feats.append(cv2.normalize(hist, hist).flatten())
    return np.concatenate(feats)

def extract_lbp_features(img_bgr, radius=2, n_points=16):
    from skimage.feature import local_binary_pattern
    gray = cv2.cvtColor(cv2.resize(img_bgr, (64,64)), cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, range=(0, n_points+2))
    return hist.astype(float) / (hist.sum() + 1e-6)

# Install skimage if not available
try:
    from skimage.feature import local_binary_pattern
    USE_LBP = True
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-image",
                    "--break-system-packages", "-q"], capture_output=True)
    try:
        from skimage.feature import local_binary_pattern
        USE_LBP = True
    except:
        USE_LBP = False

def extract_features(img_bgr):
    feats = []
    feats.append(extract_hog_features(img_bgr))
    feats.append(extract_color_histogram(img_bgr))
    if USE_LBP:
        feats.append(extract_lbp_features(img_bgr))
    return np.concatenate(feats)

features = []
labels_multi = []
labels_binary = []
crop_info = []

MAX_PER_CLASS = 600  # balance classes
class_counts = {c: 0 for c in RAW_CLASS_MAP.values()}

# Shuffle for balanced sampling
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

for _, row in df_shuffled.iterrows():
    cls = row['class']
    if class_counts[cls] >= MAX_PER_CLASS:
        continue
    try:
        img = cv2.imread(row['img_path'])
        if img is None:
            continue
        crop = img[row['ymin']:row['ymax'], row['xmin']:row['xmax']]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue
        feat = extract_features(crop)
        features.append(feat)
        labels_multi.append(cls)
        labels_binary.append(row['binary'])
        class_counts[cls] += 1
        crop_info.append(row)
    except Exception as e:
        pass

X = np.array(features)
y_multi  = np.array(labels_multi)
y_binary = np.array(labels_binary)

print(f"    Feature matrix shape: {X.shape}")
print(f"    Multi-class samples:  {len(y_multi)}")
print(f"    Binary samples:       {len(y_binary)}")
print(f"    Class counts (balanced): {dict(zip(*np.unique(y_multi, return_counts=True)))}")

# Save crop metadata
crop_df = pd.DataFrame(crop_info)
crop_df['label_multi']  = labels_multi
crop_df['label_binary'] = labels_binary
crop_df.to_csv(os.path.join(OUT_DIR, "crop_metadata.csv"), index=False)

# ─── 4. SPLIT DATA ─────────────────────────────────────────────────────────────
print("\n[4/7] Splitting train/test...")

# Multi-class split
le_multi = LabelEncoder()
y_multi_enc = le_multi.fit_transform(y_multi)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_multi_enc, test_size=0.2, random_state=42, stratify=y_multi_enc)

# Binary split
le_bin = LabelEncoder()
y_bin_enc = le_bin.fit_transform(y_binary)
X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
    X, y_bin_enc, test_size=0.2, random_state=42, stratify=y_bin_enc)

print(f"    Train: {len(X_tr)} | Test: {len(X_te)}")
print(f"    Classes: {list(le_multi.classes_)}")

# ─── 5. TRAIN MODELS ───────────────────────────────────────────────────────────
print("\n[5/7] Training models (SVM, Random Forest, Gradient Boosting)...")

results = {}

def eval_model(name, model, X_tr, X_te, y_tr, y_te, le, task="multi"):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time()-t0
    y_pred = model.predict(X_te)
    report = classification_report(y_te, y_pred,
                                   target_names=le.classes_,
                                   output_dict=True)
    cm = confusion_matrix(y_te, y_pred)
    acc = report['accuracy']
    print(f"    [{name}] acc={acc:.3f} | train={train_time:.1f}s")
    return {
        "name": name, "task": task,
        "accuracy": acc,
        "report": report,
        "confusion": cm,
        "model": model,
        "train_time": train_time,
        "y_pred": y_pred,
        "y_test": y_te,
        "label_encoder": le,
    }

# ── 5a. SVM ────────────────────────────────────────────────────────────────────
print("  [SVM] multi-class (PCA + StandardScaler + SVC)...")
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=150, random_state=42)),
    ('svc',    SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)),
])
results['SVM_multi']  = eval_model("SVM (multi)", svm_pipe, X_tr, X_te, y_tr, y_te, le_multi)

print("  [SVM] binary...")
svm_bin = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=100, random_state=42)),
    ('svc',    SVC(kernel='rbf', C=5, gamma='scale', probability=True, random_state=42)),
])
results['SVM_binary'] = eval_model("SVM (binary)", svm_bin, X_tr_b, X_te_b, y_tr_b, y_te_b, le_bin, "binary")

# ── 5b. Random Forest ─────────────────────────────────────────────────────────
print("  [RF] multi-class...")
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf',     RandomForestClassifier(n_estimators=200, max_depth=20,
                                       min_samples_split=4,
                                       n_jobs=-1, random_state=42)),
])
results['RF_multi']  = eval_model("RF (multi)", rf_pipe, X_tr, X_te, y_tr, y_te, le_multi)

print("  [RF] binary...")
rf_bin = Pipeline([
    ('scaler', StandardScaler()),
    ('rf',     RandomForestClassifier(n_estimators=200, max_depth=15,
                                       n_jobs=-1, random_state=42)),
])
results['RF_binary'] = eval_model("RF (binary)", rf_bin, X_tr_b, X_te_b, y_tr_b, y_te_b, le_bin, "binary")

# ── 5c. Gradient Boosting ─────────────────────────────────────────────────────
print("  [GBM] multi-class...")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier

# Use a faster variant: HistGradientBoosting
from sklearn.ensemble import HistGradientBoostingClassifier
gbm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=100, random_state=42)),
    ('gbm',    HistGradientBoostingClassifier(max_iter=100, max_depth=6,
                                               learning_rate=0.1, random_state=42)),
])
results['GBM_multi']  = eval_model("GBM (multi)", gbm_pipe, X_tr, X_te, y_tr, y_te, le_multi)

print("  [GBM] binary...")
gbm_bin = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=80, random_state=42)),
    ('gbm',    HistGradientBoostingClassifier(max_iter=100, max_depth=5,
                                               learning_rate=0.1, random_state=42)),
])
results['GBM_binary'] = eval_model("GBM (binary)", gbm_bin, X_tr_b, X_te_b, y_tr_b, y_te_b, le_bin, "binary")

# ─── 6. CNN ────────────────────────────────────────────────────────────────────
print("\n  [CNN] Training PyTorch CNN...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
    print(f"    PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except ImportError:
    TORCH_AVAILABLE = False
    print("    PyTorch not available, skipping CNN")

if TORCH_AVAILABLE:
    class PPEDataset(Dataset):
        def __init__(self, df_sub, transform=None):
            self.records   = df_sub.reset_index(drop=True)
            self.transform = transform
        def __len__(self):
            return len(self.records)
        def __getitem__(self, idx):
            row = self.records.iloc[idx]
            img = cv2.imread(row['img_path'])
            if img is None:
                img = np.zeros((64,64,3), dtype=np.uint8)
            crop = img[row['ymin']:row['ymax'], row['xmin']:row['xmax']]
            if crop.size == 0:
                crop = np.zeros((64,64,3), dtype=np.uint8)
            crop = cv2.cvtColor(cv2.resize(crop, (64,64)), cv2.COLOR_BGR2RGB)
            if self.transform:
                from PIL import Image
                crop = self.transform(Image.fromarray(crop))
            return crop, row['cnn_label']

    class SmallCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),   # 32x32
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),   # 16x16
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),   # 8x8
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d((4,4)),  # 4x4
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(256*4*4, 512), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )
        def forward(self, x):
            return self.classifier(self.features(x).view(x.size(0), -1))

    # Prepare CNN dataset
    crop_df2 = crop_df.copy()
    le_cnn = LabelEncoder()
    crop_df2['cnn_label'] = le_cnn.fit_transform(crop_df2['label_multi'])

    tr_idx, te_idx = train_test_split(
        range(len(crop_df2)), test_size=0.2, random_state=42,
        stratify=crop_df2['cnn_label'])

    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = PPEDataset(crop_df2.iloc[tr_idx], transform=transform_train)
    val_ds   = PPEDataset(crop_df2.iloc[te_idx],  transform=transform_val)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(le_cnn.classes_)
    model_cnn = SmallCNN(n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    EPOCHS = 25
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    print(f"    Training CNN for {EPOCHS} epochs on {device}...")
    best_val_acc = 0
    best_state = None
    t0 = time.time()

    for epoch in range(EPOCHS):
        model_cnn.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model_cnn(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(lbls)
            correct    += (out.argmax(1) == lbls).sum().item()
            total      += len(lbls)
        tr_acc = correct/total
        tr_loss = total_loss/total

        model_cnn.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out  = model_cnn(imgs)
                loss = criterion(out, lbls)
                val_loss    += loss.item()*len(lbls)
                val_correct += (out.argmax(1) == lbls).sum().item()
                val_total   += len(lbls)
        vl_acc  = val_correct/val_total
        vl_loss = val_loss/val_total

        train_accs.append(tr_acc);  val_accs.append(vl_acc)
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        scheduler.step()

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.clone() for k,v in model_cnn.state_dict().items()}

        if (epoch+1) % 5 == 0:
            print(f"      Epoch {epoch+1:>2}/{EPOCHS}  "
                  f"tr_acc={tr_acc:.3f}  val_acc={vl_acc:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.5f}")

    cnn_train_time = time.time()-t0
    print(f"    CNN training done in {cnn_train_time:.1f}s | best_val_acc={best_val_acc:.3f}")

    # Evaluate best model
    model_cnn.load_state_dict(best_state)
    model_cnn.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in val_dl:
            imgs = imgs.to(device)
            preds = model_cnn(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_lbls.extend(lbls.numpy())

    cnn_report = classification_report(all_lbls, all_preds,
                                        target_names=le_cnn.classes_,
                                        output_dict=True)
    cnn_cm = confusion_matrix(all_lbls, all_preds)
    results['CNN_multi'] = {
        "name": "CNN (multi)", "task": "multi",
        "accuracy": cnn_report['accuracy'],
        "report": cnn_report,
        "confusion": cnn_cm,
        "model": model_cnn,
        "train_time": cnn_train_time,
        "y_pred": np.array(all_preds),
        "y_test": np.array(all_lbls),
        "label_encoder": le_cnn,
        "train_accs": train_accs, "val_accs": val_accs,
        "train_losses": train_losses, "val_losses": val_losses,
    }

    # Save CNN training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.plot(train_losses, label='Train Loss', color='#e74c3c')
    ax1.plot(val_losses,   label='Val Loss',   color='#3498db')
    ax1.set_title("CNN Training Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(train_accs, label='Train Acc', color='#e74c3c')
    ax2.plot(val_accs,   label='Val Acc',   color='#3498db')
    ax2.set_title("CNN Training Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_cnn_training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ─── 7. EVALUATION & VISUALIZATION ────────────────────────────────────────────
print("\n[6/7] Generating evaluation plots...")

# 7a. Model comparison bar chart
multi_res  = {k: v for k,v in results.items() if v['task']=='multi'}
binary_res = {k: v for k,v in results.items() if v['task']=='binary'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, res_dict, title in [
    (axes[0], multi_res,  "Multi-class Accuracy"),
    (axes[1], binary_res, "Binary Accuracy"),
]:
    names = [v['name'] for v in res_dict.values()]
    accs  = [v['accuracy'] for v in res_dict.values()]
    colors_bar = ['#e74c3c','#3498db','#2ecc71','#f39c12'][:len(names)]
    bars = ax.bar(names, accs, color=colors_bar, edgecolor='black', width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis='x', rotation=20)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle("Model Comparison — PPE Detection", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# 7b. Confusion matrices (multi-class)
fig, axes = plt.subplots(1, len(multi_res), figsize=(6*len(multi_res), 5))
if len(multi_res) == 1:
    axes = [axes]
for ax, (key, res) in zip(axes, multi_res.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=res['confusion'],
        display_labels=res['label_encoder'].classes_)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(f"{res['name']}\nacc={res['accuracy']:.3f}", fontsize=10)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_confusion_matrices.png"), dpi=150, bbox_inches='tight')
plt.close()

# 7c. Per-class F1 heatmap
fig, ax = plt.subplots(figsize=(10, 4))
cls_names = list(le_multi.classes_)
f1_data = {}
for key, res in multi_res.items():
    rpt = res['report']
    f1_data[res['name']] = [rpt.get(c, {}).get('f1-score', 0) for c in cls_names]
f1_df = pd.DataFrame(f1_data, index=cls_names)
sns.heatmap(f1_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label':'F1-Score'})
ax.set_title("Per-class F1 Score by Model", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_f1_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

# 7d. Binary ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
bin_colors = ['#e74c3c','#3498db','#2ecc71']
for (key, res), col in zip(binary_res.items(), bin_colors):
    mdl = res['model']
    if hasattr(mdl, 'predict_proba'):
        y_score = mdl.predict_proba(X_te_b)[:,1]
        fpr, tpr, _ = roc_curve(y_te_b, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{res['name']} (AUC={roc_auc:.3f})")
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.05])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Binary PPE Classification", fontsize=13, fontweight='bold')
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_roc_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

# 7e. Feature importance (RF)
if 'RF_multi' in results:
    rf_model = results['RF_multi']['model'].named_steps['rf']
    importances = rf_model.feature_importances_
    top_k = 30
    top_idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_k), importances[top_idx], color='steelblue', edgecolor='black')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"feat_{i}" for i in top_idx], fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Top 30 Feature Importances", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_rf_feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ─── 8. SUMMARY TABLE ─────────────────────────────────────────────────────────
print("\n[7/7] Generating summary report...")

rows = []
for key, res in results.items():
    rpt = res['report']
    rows.append({
        'Model': res['name'],
        'Task': res['task'],
        'Accuracy': f"{res['accuracy']:.4f}",
        'Macro F1': f"{rpt.get('macro avg',{}).get('f1-score',0):.4f}",
        'Weighted F1': f"{rpt.get('weighted avg',{}).get('f1-score',0):.4f}",
        'Train Time (s)': f"{res['train_time']:.1f}",
    })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(os.path.join(OUT_DIR, "model_summary.csv"), index=False)

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(summary_df.to_string(index=False))
print("="*70)

# Save models
for key, res in results.items():
    if key != 'CNN_multi':
        joblib.dump(res['model'], os.path.join(OUT_DIR, f"model_{key}.pkl"))
if 'CNN_multi' in results and TORCH_AVAILABLE:
    torch.save(best_state, os.path.join(OUT_DIR, "model_CNN_multi.pth"))

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"  {f:45s} {sz/1024:8.1f} KB")
