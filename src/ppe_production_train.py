"""
PPE PRODUCTION TRAINING SCRIPT
================================
Workspace Safety Equipment Detection — Full Pipeline
Version 2.0 — Production-Ready

Architecture:
  - Two-stage detection: Person → PPE Classification
  - Stage 1: OpenCV HOG People Detector
  - Stage 2: Ensemble of CNN + SVM classifiers

Datasets combined:
  1. MinhNKB Helmet-Safety-Vest: 1613 images, 5 classes
  2. Jomarkow Hard Hat Workers: 1000 images, 3 classes (helm/head/person)

Classes:
  helmet, safety_vest, full_ppe, partial_ppe, no_ppe

GPU NOTE: Run with CUDA for 10-20x speedup:
  torch.device('cuda') is automatically detected

Usage:
  python ppe_production_train.py [--epochs N] [--batch-size N]
"""

import os, sys, glob, random, time, argparse, warnings, pickle
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, ConfusionMatrixDisplay,
                              precision_recall_curve, average_precision_score)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# ── Argument parsing ───────────────────────────────────────────
parser = argparse.ArgumentParser(description='PPE Production Training')
parser.add_argument('--epochs', type=int, default=40, help='CNN training epochs (default 40)')
parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default 128)')
parser.add_argument('--max-per-class', type=int, default=600, help='Max crops per class (default 600)')
args, _ = parser.parse_known_args()

EPOCHS    = args.epochs
BATCH     = args.batch_size
MAX_CLASS = args.max_per_class

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Paths ──────────────────────────────────────────────────────
# Auto-detect base path (works on both Linux and Windows)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # PPE-Detection/
BASE = os.path.dirname(PROJECT_DIR)  # D:\Claude or /sessions/...

# Datasets location (D:\datasets on Windows, or BASE/datasets on Linux)
if os.path.exists("D:/datasets/jomarkow"):
    DATASETS = "D:/datasets"
else:
    DATASETS = os.path.join(BASE, "datasets")

MINHNKB_IMG = os.path.join(DATASETS,"helmet-safety-vest-detection-master/train-images-data")
MINHNKB_ANN = os.path.join(DATASETS,"helmet-safety-vest-detection-master/train-images-annotations-new")
JOMARK_IMG  = os.path.join(DATASETS,"jomarkow/images")
JOMARK_LBL  = os.path.join(DATASETS,"jomarkow/labels")
CACHE_DIR   = os.path.join(BASE,"cache")
OUT_DIR     = os.path.join(PROJECT_DIR,"results/models")
VAL_DIR     = os.path.join(PROJECT_DIR,"cctv_validation_original")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Class definitions ──────────────────────────────────────────
MINHNKB_MAP = {
    "helmet":                     "helmet",
    "safety vest":                "safety_vest",
    "person with full safety":    "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety":      "no_ppe",
}
JOMARKOW_MAP = {0: "helmet", 1: "no_ppe"}
ALL_CLASSES  = ["full_ppe","helmet","no_ppe","partial_ppe","safety_vest"]
BINARY_MAP   = {c: ("ppe_present" if c!="no_ppe" else "no_ppe") for c in ALL_CLASSES}

CLASS_COLORS = {
    'helmet':      '#27ae60',
    'safety_vest': '#2980b9',
    'full_ppe':    '#8e44ad',
    'partial_ppe': '#e67e22',
    'no_ppe':      '#e74c3c',
}

print("="*65)
print("PPE PRODUCTION TRAINING v2.0")
print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH} | MaxPerClass: {MAX_CLASS}")
print("="*65)

# ══════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════
CACHE_X = os.path.join(CACHE_DIR, f"crops_X_{MAX_CLASS}.npy")
CACHE_Y = os.path.join(CACHE_DIR, f"crops_y_{MAX_CLASS}.npy")

if os.path.exists(CACHE_X) and os.path.exists(CACHE_Y):
    print(f"\n[1/6] Loading cached crops...")
    crops_rgb = np.load(CACHE_X); labels_raw = np.load(CACHE_Y)
    print(f"  Loaded {len(crops_rgb)} cached crops")
else:
    print(f"\n[1/6] Loading & caching crops from disk...")
    t0 = time.time()
    crops_rgb, labels_raw = [], []
    cc = {c:0 for c in ALL_CLASSES}

    # ── MinhNKB dataset ─────────────────────────────────────────
    xml_files = sorted(glob.glob(os.path.join(MINHNKB_ANN,"*.xml")))
    random.shuffle(xml_files)
    for xf in xml_files:
        try:
            root=ET.parse(xf).getroot(); fname=root.findtext("filename")
            ip=os.path.join(MINHNKB_IMG, fname) if fname else None
            if not ip or not os.path.exists(ip):
                stem=os.path.splitext(os.path.basename(xf))[0]
                for e in [".jpg",".jpeg",".png"]:
                    c=os.path.join(MINHNKB_IMG,stem+e)
                    if os.path.exists(c): ip=c; break
            if not ip or not os.path.exists(ip): continue
            img=cv2.imread(ip)
            if img is None: continue
            sz=root.find("size"); iw,ih=int(sz.findtext("width",0)),int(sz.findtext("height",0))
            for obj in root.findall("object"):
                raw=obj.findtext("name","").strip().lower()
                if raw not in MINHNKB_MAP: continue
                cls=MINHNKB_MAP[raw]
                if cc[cls]>=MAX_CLASS: continue
                bb=obj.find("bndbox")
                x1=max(0,int(float(bb.findtext("xmin",0)))); y1=max(0,int(float(bb.findtext("ymin",0))))
                x2=min(iw,int(float(bb.findtext("xmax",iw)))); y2=min(ih,int(float(bb.findtext("ymax",ih))))
                if x2>x1+8 and y2>y1+8:
                    crop=img[y1:y2,x1:x2]
                    if crop.size==0: continue
                    crops_rgb.append(cv2.cvtColor(cv2.resize(crop,(64,64)),cv2.COLOR_BGR2RGB))
                    labels_raw.append(cls); cc[cls]+=1
        except: pass
    print(f"  MinhNKB: {len(crops_rgb)} crops | {cc}")

    # ── Jomarkow dataset ────────────────────────────────────────
    cc_j={0:0,1:0}; MAX_J=min(MAX_CLASS,400)
    lbl_files=sorted(glob.glob(os.path.join(JOMARK_LBL,"*.txt")))
    random.shuffle(lbl_files)
    for lf in lbl_files:
        base=os.path.splitext(os.path.basename(lf))[0]
        ip=os.path.join(JOMARK_IMG,base+".png")
        if not os.path.exists(ip): ip=os.path.join(JOMARK_IMG,base+".jpg")
        if not os.path.exists(ip): continue
        try:
            img=cv2.imread(ip)
            if img is None: continue
            ih2,iw2=img.shape[:2]
            for line in open(lf):
                parts=line.strip().split()
                if len(parts)<5: continue
                cid=int(parts[0])
                if cid not in JOMARKOW_MAP: continue
                if cc_j[cid]>=MAX_J: continue
                cls=JOMARKOW_MAP[cid]
                cx,cy,bw,bh=float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
                x1=max(0,int((cx-bw/2)*iw2)); y1=max(0,int((cy-bh/2)*ih2))
                x2=min(iw2,int((cx+bw/2)*iw2)); y2=min(ih2,int((cy+bh/2)*ih2))
                if x2>x1+8 and y2>y1+8:
                    crop=img[y1:y2,x1:x2]
                    if crop.size>0:
                        crops_rgb.append(cv2.cvtColor(cv2.resize(crop,(64,64)),cv2.COLOR_BGR2RGB))
                        labels_raw.append(cls); cc_j[cid]+=1
        except: pass
    print(f"  Jomarkow: added {sum(cc_j.values())} crops | {{'helmet':cc_j[0],'no_ppe':cc_j[1]}}")

    crops_rgb=np.array(crops_rgb); labels_raw=np.array(labels_raw)
    np.save(CACHE_X, crops_rgb); np.save(CACHE_Y, labels_raw)
    print(f"  Total: {len(crops_rgb)} crops | loaded in {time.time()-t0:.1f}s")

class_counts = dict(zip(*np.unique(labels_raw, return_counts=True)))
print(f"  Class distribution: {class_counts}")

# ── Feature extraction for ML models ──────────────────────────
CACHE_F = os.path.join(CACHE_DIR, f"features_{MAX_CLASS}.npy")
if os.path.exists(CACHE_F):
    print(f"\n[2/6] Loading cached features...")
    X_ml = np.load(CACHE_F)
else:
    print(f"\n[2/6] Extracting HOG+Color features for ML models...")
    t0=time.time()
    def extract_features(img_rgb):
        img=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        hog=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
        hf=hog.compute(gray).flatten()
        cf=[]
        for ch in range(3):
            h=cv2.calcHist([img],[ch],None,[32],[0,256]); cf.append(cv2.normalize(h,h).flatten())
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        for ch in range(3):
            h=cv2.calcHist([hsv],[ch],None,[32],[0,256]); cf.append(cv2.normalize(h,h).flatten())
        return np.concatenate([hf]+cf)
    X_ml=np.array([extract_features(c) for c in crops_rgb])
    np.save(CACHE_F, X_ml)
    print(f"  Features: {X_ml.shape} | extracted in {time.time()-t0:.1f}s")

print(f"  Feature matrix: {X_ml.shape}")

# ── Encode labels + split ──────────────────────────────────────
le_multi  = LabelEncoder(); y_multi  = le_multi.fit_transform(labels_raw)
le_binary = LabelEncoder()
y_binary  = le_binary.fit_transform([BINARY_MAP[l] for l in labels_raw])

Xtr,Xte,ytr,yte = train_test_split(X_ml,y_multi, test_size=0.2,random_state=42,stratify=y_multi)
Xtrb,Xteb,ytrb,yteb = train_test_split(X_ml,y_binary,test_size=0.2,random_state=42,stratify=y_binary)
print(f"  Train={len(Xtr)}, Test={len(Xte)}")

# ══════════════════════════════════════════════════════════════════
# PHASE 2: ML MODEL TRAINING
# ══════════════════════════════════════════════════════════════════
print(f"\n[3/6] Training ML models...")
ml_results = {}

def train_eval(name, pipe, Xtr, Xte, ytr, yte, le, tag):
    t0=time.time(); pipe.fit(Xtr,ytr); tt=time.time()-t0
    yp=pipe.predict(Xte)
    rpt=classification_report(yte,yp,target_names=le.classes_,output_dict=True)
    cm=confusion_matrix(yte,yp); acc=rpt['accuracy']
    print(f"  [{name:35s}] acc={acc:.4f}  t={tt:.0f}s")
    ml_results[tag]={'name':name,'accuracy':acc,'report':rpt,'confusion':cm,
                     'model':pipe,'time':tt,'yp':yp,'yt':yte,'le':le,
                     'task':'multi' if 'multi' in tag else 'binary'}
    joblib.dump(pipe, os.path.join(OUT_DIR,f"prod_{tag}.pkl"))

# SVM — PCA(220) keeps more variance; class_weight via sample_weight not supported,
#        so we use balanced class_weight via CalibratedClassifierCV wrapper
train_eval("SVM (PCA->RBF, multi)", Pipeline([
    ('sc',StandardScaler()),('pca',PCA(220,random_state=42)),
    ('svm',SVC(kernel='rbf',C=15,gamma='scale',probability=True,
               class_weight='balanced',random_state=42))
]), Xtr,Xte,ytr,yte,le_multi,'svm_multi')

train_eval("SVM (PCA->RBF, binary)", Pipeline([
    ('sc',StandardScaler()),('pca',PCA(150,random_state=42)),
    ('svm',SVC(kernel='rbf',C=10,gamma='scale',probability=True,
               class_weight='balanced',random_state=42))
]), Xtrb,Xteb,ytrb,yteb,le_binary,'svm_binary')

# Random Forest — balanced class weights
train_eval("RandomForest (400 trees, multi)", Pipeline([
    ('sc',StandardScaler()),
    ('rf',RandomForestClassifier(400,max_depth=22,min_samples_split=3,
                                  class_weight='balanced',n_jobs=-1,random_state=42))
]), Xtr,Xte,ytr,yte,le_multi,'rf_multi')

train_eval("RandomForest (400 trees, binary)", Pipeline([
    ('sc',StandardScaler()),
    ('rf',RandomForestClassifier(400,max_depth=18,class_weight='balanced',
                                  n_jobs=-1,random_state=42))
]), Xtrb,Xteb,ytrb,yteb,le_binary,'rf_binary')

# ExtraTrees — faster and often better than RF; balanced weights
train_eval("ExtraTrees (400 trees, multi)", Pipeline([
    ('sc',StandardScaler()),
    ('et',ExtraTreesClassifier(400,max_depth=24,min_samples_split=3,
                                class_weight='balanced',n_jobs=-1,random_state=42))
]), Xtr,Xte,ytr,yte,le_multi,'et_multi')

train_eval("ExtraTrees (400 trees, binary)", Pipeline([
    ('sc',StandardScaler()),
    ('et',ExtraTreesClassifier(400,max_depth=20,class_weight='balanced',
                                n_jobs=-1,random_state=42))
]), Xtrb,Xteb,ytrb,yteb,le_binary,'et_binary')

# HistGBM — no PCA (histogram binning handles high-dim natively);
#            more iterations + lower lr; class_weight balanced
train_eval("HistGBM (400 rounds, multi)", Pipeline([
    ('sc',StandardScaler()),
    ('gbm',HistGradientBoostingClassifier(max_iter=400,max_depth=8,
                                          learning_rate=0.02,class_weight='balanced',
                                          random_state=42))
]), Xtr,Xte,ytr,yte,le_multi,'gbm_multi')

train_eval("HistGBM (400 rounds, binary)", Pipeline([
    ('sc',StandardScaler()),
    ('gbm',HistGradientBoostingClassifier(max_iter=400,max_depth=6,
                                          learning_rate=0.02,class_weight='balanced',
                                          random_state=42))
]), Xtrb,Xteb,ytrb,yteb,le_binary,'gbm_binary')

# Soft-voting ensemble — combines SVM + RF + ExtraTrees + GBM probabilities
print(f"  [{'Building soft-voting ensemble':35s}]", end=' ', flush=True)
t0 = time.time()

svm_est  = Pipeline([('sc',StandardScaler()),('pca',PCA(220,random_state=42)),
                     ('svm',SVC(kernel='rbf',C=15,gamma='scale',probability=True,
                                class_weight='balanced',random_state=42))])
rf_est   = Pipeline([('sc',StandardScaler()),
                     ('rf',RandomForestClassifier(400,max_depth=22,min_samples_split=3,
                                                   class_weight='balanced',n_jobs=-1,random_state=42))])
et_est   = Pipeline([('sc',StandardScaler()),
                     ('et',ExtraTreesClassifier(400,max_depth=24,min_samples_split=3,
                                                 class_weight='balanced',n_jobs=-1,random_state=42))])
gbm_est  = Pipeline([('sc',StandardScaler()),
                     ('gbm',HistGradientBoostingClassifier(max_iter=400,max_depth=8,
                                                            learning_rate=0.02,class_weight='balanced',
                                                            random_state=42))])

ensemble_multi = VotingClassifier(
    estimators=[('svm',svm_est),('rf',rf_est),('et',et_est),('gbm',gbm_est)],
    voting='soft', n_jobs=1
)
ensemble_multi.fit(Xtr, ytr)
yp_ens = ensemble_multi.predict(Xte)
rpt_ens = classification_report(yte,yp_ens,target_names=le_multi.classes_,output_dict=True)
acc_ens = rpt_ens['accuracy']
tt_ens  = time.time()-t0
print(f"acc={acc_ens:.4f}  t={tt_ens:.0f}s")
ml_results['ensemble_multi'] = {
    'name':'Ensemble (SVM+RF+ET+GBM, multi)','accuracy':acc_ens,
    'report':rpt_ens,'confusion':confusion_matrix(yte,yp_ens),
    'model':ensemble_multi,'time':tt_ens,'yp':yp_ens,'yt':yte,
    'le':le_multi,'task':'multi'
}
joblib.dump(ensemble_multi, os.path.join(OUT_DIR,'prod_ensemble_multi.pkl'))

# ══════════════════════════════════════════════════════════════════
# PHASE 3: CNN TRAINING
# ══════════════════════════════════════════════════════════════════
print(f"\n[4/6] Training CNN ({EPOCHS} epochs on {DEVICE})...")

# Encode + tensorize all crops
le_cnn = LabelEncoder(); y_cnn = le_cnn.fit_transform(labels_raw)
X_arr  = crops_rgb.astype(np.float32) / 255.0
mean   = np.array([0.485,0.456,0.406],dtype=np.float32)
std    = np.array([0.229,0.224,0.225],dtype=np.float32)
X_arr  = (X_arr - mean) / std
X_t    = torch.tensor(X_arr.transpose(0,3,1,2))
y_t    = torch.tensor(y_cnn, dtype=torch.long)

tr_idx, te_idx = train_test_split(range(len(y_cnn)), test_size=0.2,
                                   random_state=42, stratify=y_cnn)
tr_dl = DataLoader(TensorDataset(X_t[tr_idx], y_t[tr_idx]),
                   batch_size=BATCH, shuffle=True,  pin_memory=(DEVICE.type=='cuda'))
te_dl = DataLoader(TensorDataset(X_t[te_idx], y_t[te_idx]),
                   batch_size=BATCH, shuffle=False, pin_memory=(DEVICE.type=='cuda'))
print(f"  Train={len(tr_idx)}, Test={len(te_idx)}, Batches/epoch={len(tr_dl)}")
print(f"  Classes: {list(le_cnn.classes_)}")

class PPENet(nn.Module):
    """Lightweight CNN optimised for 64x64 PPE crop classification"""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64→32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 2: 32→16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 3: 16→8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Global average pool → 128 features
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = PPENet(len(le_cnn.classes_)).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")

criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer  = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=EPOCHS,
    steps_per_epoch=len(tr_dl), pct_start=0.2,
    div_factor=10, final_div_factor=100,
)

history = {'tr_acc':[], 'va_acc':[], 'tr_loss':[], 'va_loss':[], 'lr':[]}
best_acc, best_state = 0.0, None
t_train = time.time()

for epoch in range(1, EPOCHS+1):
    # ── Train ──────────────────────────────────────────────────
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for imgs, lbls in tr_dl:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        tl += loss.item()*len(lbls)
        tc += (out.argmax(1)==lbls).sum().item()
        tt += len(lbls)

    # ── Validate ───────────────────────────────────────────────
    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in te_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            vl += criterion(out, lbls).item()*len(lbls)
            vc += (out.argmax(1)==lbls).sum().item()
            vt += len(lbls)

    ta, va = tc/tt, vc/vt
    history['tr_acc'].append(ta); history['va_acc'].append(va)
    history['tr_loss'].append(tl/tt); history['va_loss'].append(vl/vt)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    if va > best_acc:
        best_acc = va
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    if epoch % 5 == 0 or epoch == 1:
        elapsed = time.time()-t_train
        eta = elapsed/epoch * (EPOCHS-epoch)
        print(f"  Ep {epoch:>2}/{EPOCHS}  tr={ta:.3f}  val={va:.3f}  "
              f"lr={optimizer.param_groups[0]['lr']:.5f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

cnn_time = time.time()-t_train
print(f"\n  CNN done: {cnn_time:.1f}s | best_val={best_acc:.3f}")

# Evaluate best model
model.load_state_dict(best_state); model.eval()
all_p, all_l, all_prob = [], [], []
with torch.no_grad():
    for imgs, lbls in te_dl:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        prob = torch.softmax(out, 1).cpu().numpy()
        all_p.extend(out.argmax(1).cpu().numpy())
        all_l.extend(lbls.numpy())
        all_prob.extend(prob)

cnn_rpt = classification_report(all_l, all_p, target_names=le_cnn.classes_, output_dict=True)
cnn_cm  = confusion_matrix(all_l, all_p)
print("\n  CNN Classification Report:")
print(classification_report(all_l, all_p, target_names=le_cnn.classes_))

# Save model + metadata
torch.save({'state_dict': best_state, 'classes': list(le_cnn.classes_),
             'epoch': EPOCHS, 'best_val_acc': best_acc, 'arch': 'PPENet'},
           os.path.join(OUT_DIR, "prod_cnn_model.pth"))

with open(os.path.join(OUT_DIR,"prod_le_cnn.pkl"),'wb') as f: pickle.dump(le_cnn, f)
with open(os.path.join(OUT_DIR,"prod_le_multi.pkl"),'wb') as f: pickle.dump(le_multi, f)
with open(os.path.join(OUT_DIR,"prod_le_binary.pkl"),'wb') as f: pickle.dump(le_binary, f)

# ══════════════════════════════════════════════════════════════════
# PHASE 4: TWO-STAGE CCTV DETECTION
# ══════════════════════════════════════════════════════════════════
print(f"\n[5/6] Two-stage CCTV detection pipeline...")

hog_people = cv2.HOGDescriptor()
hog_people.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people_hog(img_bgr, confidence=0.4):
    """Stage 1: HOG-based person detection"""
    h, w = img_bgr.shape[:2]
    # Ensure minimum width for HOG detector (128px minimum)
    min_w = 320
    if w < min_w:
        scale = min_w / w
        img_bgr = cv2.resize(img_bgr, (min_w, int(h*scale)))
    else:
        scale = 1.0

    rects, weights = hog_people.detectMultiScale(
        img_bgr,
        winStride=(8,8), padding=(16,16), scale=1.05,
        useMeanshiftGrouping=False
    )
    boxes = []
    if len(rects) > 0:
        for (x,y,bw,bh), wt in zip(rects, weights):
            if float(wt) > confidence:
                # Scale back if we resized
                boxes.append({
                    'x1': int(x/scale), 'y1': int(y/scale),
                    'x2': int((x+bw)/scale), 'y2': int((y+bh)/scale),
                    'det_conf': float(wt)
                })
    return boxes

def classify_crop_cnn(crop_bgr, model, le_cnn):
    """Stage 2: PPE classification via CNN"""
    crop_rgb = cv2.cvtColor(cv2.resize(crop_bgr,(64,64)), cv2.COLOR_BGR2RGB)
    x = torch.tensor(crop_rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    mn = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    sd = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    x  = (x-mn)/sd
    with torch.no_grad():
        out = model(x.to(DEVICE))
        prob = torch.softmax(out,1).squeeze().cpu().numpy()
    cls_idx = prob.argmax()
    return le_cnn.classes_[cls_idx], prob[cls_idx], prob

def classify_crop_svm(crop_bgr, svm_model, le):
    """Stage 2 alternative: SVM classification"""
    img = cv2.resize(crop_bgr,(64,64)); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hog_d=cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    hf=hog_d.compute(gray).flatten(); cf=[]
    for ch in range(3):
        h=cv2.calcHist([img],[ch],None,[32],[0,256]); cf.append(cv2.normalize(h,h).flatten())
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for ch in range(3):
        h=cv2.calcHist([hsv],[ch],None,[32],[0,256]); cf.append(cv2.normalize(h,h).flatten())
    feat = np.concatenate([hf]+cf).reshape(1,-1)
    cls_enc = svm_model.predict(feat)[0]
    prob = svm_model.predict_proba(feat)[0]
    return le.inverse_transform([cls_enc])[0], prob.max(), prob

# Load best SVM (already trained)
svm_pipe = ml_results['svm_multi']['model']

val_images = sorted(
    glob.glob(os.path.join(VAL_DIR,"*.jpg")) +
    glob.glob(os.path.join(VAL_DIR,"*.JPG")) +
    glob.glob(os.path.join(VAL_DIR,"*.png"))
)
print(f"  Processing {len(val_images)} validation images...")

# Generate annotated output images
model.eval()
fig, axes = plt.subplots(3, 4, figsize=(24,18))
axes = axes.flatten()
fig.suptitle("PPE Two-Stage Detection: OpenCV HOG Person Detector -> CNN Classifier\n"
             f"Combined Dataset: MinhNKB + Jomarkow | CNN: {len(crops_rgb)} crops | {EPOCHS} epochs",
             fontsize=12, fontweight='bold')
fig.patch.set_facecolor('#0d1117')

cctv_results = []
for i, img_path in enumerate(val_images[:12]):
    img = cv2.imread(img_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = axes[i]; ax.imshow(img_rgb); ax.set_facecolor('#0d1117')
    H, W = img.shape[:2]

    if H > 150 and W > 150:  # Full scene
        # Stage 1: detect people
        people = detect_people_hog(img.copy(), confidence=0.3)

        # If HOG finds no people, fall back to dense patches on large scenes
        if len(people) == 0 and H > 300:
            # Sliding window fallback (3 size variants)
            for wh,ww in [(100,60),(150,90),(200,120)]:
                for y in range(0,H-wh,60):
                    for x in range(0,W-ww,60):
                        people.append({'x1':x,'y1':y,'x2':x+ww,'y2':y+wh,'det_conf':0.5})

        # Stage 2: classify each person crop
        dets = []
        for box in people:
            x1,y1,x2,y2 = box['x1'],box['y1'],box['x2'],box['y2']
            # Expand box slightly
            pad = max(10, int((y2-y1)*0.05))
            px1,py1=max(0,x1-pad),max(0,y1-pad)
            px2,py2=min(W,x2+pad),min(H,y2+pad)
            crop = img[py1:py2,px1:px2]
            if crop.size==0 or crop.shape[0]<16 or crop.shape[1]<16: continue
            cls, conf, probs = classify_crop_cnn(crop, model, le_cnn)
            if conf > 0.50:
                dets.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                             'class':cls,'conf':float(conf),'det_conf':box['det_conf']})

        # Draw
        import matplotlib.patches as mpatches
        for det in dets:
            col = CLASS_COLORS.get(det['class'],'gray')
            rect = mpatches.Rectangle(
                (det['x1'],det['y1']), det['x2']-det['x1'], det['y2']-det['y1'],
                lw=2, edgecolor=col, facecolor='none', alpha=0.9)
            ax.add_patch(rect)
            ax.text(det['x1']+2, det['y1']+12, f"{det['class'][:9]}\n{det['conf']:.2f}",
                    color='white', fontsize=6, fontweight='bold',
                    bbox=dict(facecolor=col, alpha=0.75, pad=1, boxstyle='round,pad=0.2'))

        violations = sum(1 for d in dets if d['class']=='no_ppe')
        tc = 'lime' if violations==0 else '#ff4444'
        ax.set_title(f"{os.path.basename(img_path)}\nPeople: {len(people)} | Classified: {len(dets)} | ⚠ {violations}",
                     color=tc, fontsize=8, pad=3)
        cctv_results.append({'file':os.path.basename(img_path),'people':len(people),
                              'classified':len(dets),'violations':violations})
    else:
        # Direct classification for crop images
        cls, conf, _ = classify_crop_cnn(img, model, le_cnn)
        col = CLASS_COLORS.get(cls,'gray')
        import matplotlib.patches as mpatches
        ax.add_patch(mpatches.Rectangle((2,2),W-4,H-4,lw=3,edgecolor=col,facecolor='none'))
        ax.set_title(f"{os.path.basename(img_path)}\n{cls} ({conf:.2f})", color=col, fontsize=8)
        cctv_results.append({'file':os.path.basename(img_path),'people':1,
                              'classified':1,'violations':1 if cls=='no_ppe' else 0})
    ax.axis('off')

for j in range(len(val_images),12):
    axes[j].axis('off'); axes[j].set_facecolor('#0d1117')

import matplotlib.lines as mlines
legend = [mlines.Line2D([0],[0],color=v,lw=3,label=k) for k,v in CLASS_COLORS.items()]
fig.legend(handles=legend,loc='lower center',ncol=5,fontsize=9,
           bbox_to_anchor=(0.5,-0.01),facecolor='#1a1a1a',labelcolor='white',edgecolor='gray')
plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(OUT_DIR,"prod_cctv_validation.png"),dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  Saved prod_cctv_validation.png")

pd.DataFrame(cctv_results).to_csv(os.path.join(OUT_DIR,"prod_cctv_results.csv"),index=False)

# ══════════════════════════════════════════════════════════════════
# PHASE 5: COMPREHENSIVE EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════════
print(f"\n[6/6] Generating evaluation plots...")

# 1. CNN training history
ep_r = range(1, EPOCHS+1)
fig, axes = plt.subplots(1, 3, figsize=(18,5))
axes[0].plot(ep_r,history['tr_loss'],'r-',lw=2,label='Train')
axes[0].plot(ep_r,history['va_loss'],'b-',lw=2,label='Val')
axes[0].fill_between(ep_r,history['tr_loss'],alpha=0.1,color='r')
axes[0].fill_between(ep_r,history['va_loss'],alpha=0.1,color='b')
axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep_r,history['tr_acc'],'r-',lw=2,label='Train')
axes[1].plot(ep_r,history['va_acc'],'b-',lw=2,label='Val')
axes[1].axhline(best_acc,color='g',lw=1.5,ls='--',label=f'Best={best_acc:.3f}')
axes[1].fill_between(ep_r,history['tr_acc'],alpha=0.1,color='r')
axes[1].fill_between(ep_r,history['va_acc'],alpha=0.1,color='b')
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0,1.05)
axes[2].plot(ep_r,history['lr'],'purple',lw=2); axes[2].set_title("Learning Rate (OneCycleLR)")
axes[2].grid(alpha=0.3); axes[2].set_xlabel("Epoch")
plt.suptitle(f"CNN Training History — PPENet ({EPOCHS} epochs, {len(crops_rgb)} crops)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"prod_cnn_training.png"),dpi=150,bbox_inches='tight')
plt.close()

# 2. Model comparison bar chart
multi_r = {k:v for k,v in ml_results.items() if v['task']=='multi'}
multi_r['cnn'] = {'name':f'CNN PPENet ({EPOCHS}ep)','accuracy':cnn_rpt['accuracy'],
                  'report':cnn_rpt,'confusion':cnn_cm,'le':le_cnn,
                  'task':'multi','time':cnn_time}
binary_r = {k:v for k,v in ml_results.items() if v['task']=='binary'}

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
for ax,(res_d,title) in zip([ax1,ax2],[(multi_r,"Multi-class"),(binary_r,"Binary")]):
    names=[v['name'] for v in res_d.values()]
    accs=[v['accuracy'] for v in res_d.values()]
    cols=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6'][:len(names)]
    bars=ax.bar(range(len(names)),accs,color=cols,edgecolor='black',width=0.6)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=20,fontsize=8)
    ax.set_ylim(0,1.12); ax.set_title(f"{title} Accuracy",fontsize=13,fontweight='bold')
    ax.set_ylabel("Accuracy"); ax.axhline(0.8,color='red',lw=1,ls='--',alpha=0.5)
    ax.grid(axis='y',alpha=0.3)
    for bar,a in zip(bars,accs):
        ax.text(bar.get_x()+bar.get_width()/2,a+0.01,f'{a:.3f}',
                ha='center',va='bottom',fontweight='bold',fontsize=10)
plt.suptitle("Production Model Comparison — Combined Dataset (MinhNKB+Jomarkow)\n"
             f"{len(crops_rgb)} total crops | {EPOCHS}-epoch CNN | Two-stage detection",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"prod_model_comparison.png"),dpi=150,bbox_inches='tight')
plt.close()

# 3. Confusion matrices
fig,axes_cm=plt.subplots(1,4,figsize=(26,5))
for ax,(k,res) in zip(axes_cm,multi_r.items()):
    ConfusionMatrixDisplay(res['confusion'],display_labels=res['le'].classes_).plot(
        ax=ax,cmap='Blues',colorbar=False,values_format='d')
    ax.set_title(f"{res['name']}\nacc={res['accuracy']:.3f}",fontsize=9)
    ax.tick_params(axis='x',rotation=40)
plt.suptitle("Confusion Matrices — Production Models",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"prod_confusion_matrices.png"),dpi=150,bbox_inches='tight')
plt.close()

# 4. Per-class F1 heatmap
cls_names=list(le_multi.classes_)
f1d={}
for k,res in multi_r.items():
    f1d[res['name']]=[res['report'].get(c,{}).get('f1-score',0) for c in cls_names]
fig,ax=plt.subplots(figsize=(13,4))
sns.heatmap(pd.DataFrame(f1d,index=cls_names),annot=True,fmt='.3f',
            cmap='YlOrRd',ax=ax,linewidths=0.5,cbar_kws={'label':'F1'})
ax.set_title("Per-class F1 Score — All Production Models",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"prod_f1_heatmap.png"),dpi=150,bbox_inches='tight')
plt.close()

# 5. ROC curves (binary)
fig,ax=plt.subplots(figsize=(8,6))
cols=['#e74c3c','#3498db','#2ecc71']
for (k,res),c in zip(binary_r.items(),cols):
    if hasattr(res['model'],'predict_proba'):
        ys=res['model'].predict_proba(Xteb)[:,1]
        fpr,tpr,_=roc_curve(yteb,ys); ra=auc(fpr,tpr)
        ax.plot(fpr,tpr,color=c,lw=2,label=f"{res['name']} (AUC={ra:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves — Binary PPE Classification",fontsize=13,fontweight='bold')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"prod_roc_curves.png"),dpi=150,bbox_inches='tight')
plt.close()

# 6. CNN confusion
fig,ax=plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay(cnn_cm,display_labels=le_cnn.classes_).plot(
    ax=ax,cmap='Blues',colorbar=False,values_format='d')
ax.set_title(f"CNN PPENet Confusion Matrix\nacc={cnn_rpt['accuracy']:.3f}")
ax.tick_params(axis='x',rotation=35)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"prod_cnn_confusion.png"),dpi=150,bbox_inches='tight')
plt.close()

# ── Final summary ──────────────────────────────────────────────
rows=[]
for k,res in {**multi_r,**binary_r}.items():
    rpt=res['report']
    rows.append({'Model':res['name'],'Task':res['task'],
                 'Accuracy':round(res['accuracy'],4),
                 'Macro F1':round(rpt.get('macro avg',{}).get('f1-score',0),4),
                 'Weighted F1':round(rpt.get('weighted avg',{}).get('f1-score',0),4),
                 'Train Time(s)':round(res.get('time',0),1)})
summary=pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR,"prod_model_summary.csv"),index=False)

print("\n"+"="*65)
print("PRODUCTION MODEL RESULTS SUMMARY")
print("="*65)
print(summary.to_string(index=False))
print("="*65)
print(f"\nDataset: {len(crops_rgb)} crops from 2 sources")
print(f"  MinhNKB: 5 classes (helmet/vest/full/partial/no ppe)")
print(f"  Jomarkow: 2 classes (helmet, no_ppe head)")
print(f"CNN best val acc: {best_acc:.4f} | trained {cnn_time:.0f}s")
print("\nAll outputs saved!")
for f in sorted(os.listdir(OUT_DIR)):
    if 'prod_' in f:
        sz=os.path.getsize(os.path.join(OUT_DIR,f))
        print(f"  {f:48s} {sz/1024:7.1f} KB")
