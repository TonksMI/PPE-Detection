"""
PPE Combined Pipeline — Multi-Dataset Training + Two-Stage Detection
=====================================================================
Datasets:
  1. MinhNKB: 1613 images, Pascal VOC, 5 classes
  2. Jomarkow: 1000 images, YOLO format, 3 classes
     class0=helmet, class1=head(no helmet), class2=person

Two-stage pipeline:
  Stage 1 → OpenCV HOG person detector (find people)
  Stage 2 → PPE classifier on each person crop

Models: SVM, Random Forest, HistGBM, CNN (25 epochs)
"""
import os, sys, glob, random, time, warnings, json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ── Config ─────────────────────────────────────────────────────
# Auto-detect paths (works on both Linux and Windows)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # PPE-Detection/
BASE = os.path.dirname(PROJECT_DIR)  # D:\Claude or /sessions/...

MINHNKB_IMG_DIR = os.path.join(BASE, "datasets/helmet-safety-vest-detection-master/train-images-data")
MINHNKB_ANN_DIR = os.path.join(BASE, "datasets/helmet-safety-vest-detection-master/train-images-annotations-new")
JOMARKOW_IMG_DIR = os.path.join(BASE, "datasets/jomarkow/images")
JOMARKOW_LBL_DIR = os.path.join(BASE, "datasets/jomarkow/labels")
OUT_DIR   = os.path.join(PROJECT_DIR, "results/models")
VAL_DIR   = os.path.join(BASE, "cctv_validation")
CROP_SIZE = (64, 64)
MAX_PER_CLASS_MINHNKB  = 500   # per class from MinhNKB
MAX_PER_CLASS_JOMARKOW = 400   # per class from Jomarkow
os.makedirs(OUT_DIR, exist_ok=True)

# ── Class mapping ───────────────────────────────────────────────
MINHNKB_CLASS_MAP = {
    "helmet":                     "helmet",
    "safety vest":                "safety_vest",
    "person with full safety":    "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety":      "no_ppe",
}
JOMARKOW_CLASS_MAP = {0: "helmet", 1: "no_ppe"}   # skip class 2 (full person, ambiguous)
BINARY_MAP = {c: ("ppe_present" if c != "no_ppe" else "no_ppe") for c in
              list(MINHNKB_CLASS_MAP.values()) + list(JOMARKOW_CLASS_MAP.values())}
ALL_CLASSES = ["full_ppe","helmet","no_ppe","partial_ppe","safety_vest"]

print("="*65)
print("PPE COMBINED PIPELINE — Multi-dataset + Two-Stage Detection")
print("="*65)

# ─────────────────────────────────────────────────────────────────
# SECTION 1: PARSE DATASETS
# ─────────────────────────────────────────────────────────────────
def extract_features(img_bgr):
    """HOG + color histogram features (1956-dim)"""
    img = cv2.resize(img_bgr, CROP_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    hog_f = hog.compute(gray).flatten()
    color_f = []
    for ch in range(3):
        h = cv2.calcHist([img],[ch],None,[32],[0,256])
        color_f.append(cv2.normalize(h,h).flatten())
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ch in range(3):
        h = cv2.calcHist([hsv],[ch],None,[32],[0,256])
        color_f.append(cv2.normalize(h,h).flatten())
    return np.concatenate([hog_f]+color_f)

def augment_crop(crop_bgr):
    """Return list of augmented variants (data augmentation)"""
    variants = [crop_bgr]
    # Horizontal flip
    variants.append(cv2.flip(crop_bgr, 1))
    # Brightness adjustment
    for alpha in [0.7, 1.3]:
        v = np.clip(crop_bgr.astype(np.float32)*alpha, 0, 255).astype(np.uint8)
        variants.append(v)
    # Rotation ±15°
    h,w = crop_bgr.shape[:2]
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
        variants.append(cv2.warpAffine(crop_bgr, M, (w,h)))
    return variants  # 6 variants per crop

print("\n[1/5] Parsing MinhNKB dataset...")
records_minhkb = []
xml_files = sorted(glob.glob(os.path.join(MINHNKB_ANN_DIR,"*.xml")))
class_counts_m = {c:0 for c in MINHNKB_CLASS_MAP.values()}

for xf in xml_files:
    try:
        root = ET.parse(xf).getroot()
        fname = root.findtext("filename")
        ip = os.path.join(MINHNKB_IMG_DIR, fname) if fname else None
        if not ip or not os.path.exists(ip):
            stem = os.path.splitext(os.path.basename(xf))[0]
            for e in [".jpg",".jpeg",".png"]:
                c = os.path.join(MINHNKB_IMG_DIR,stem+e)
                if os.path.exists(c): ip=c; break
        if not ip or not os.path.exists(ip): continue
        sz = root.find("size"); iw,ih = int(sz.findtext("width",0)), int(sz.findtext("height",0))
        for obj in root.findall("object"):
            raw = obj.findtext("name","").strip().lower()
            if raw not in MINHNKB_CLASS_MAP: continue
            cls = MINHNKB_CLASS_MAP[raw]
            if class_counts_m[cls] >= MAX_PER_CLASS_MINHNKB: continue
            bb = obj.find("bndbox")
            x1=max(0,int(float(bb.findtext("xmin",0)))); y1=max(0,int(float(bb.findtext("ymin",0))))
            x2=min(iw,int(float(bb.findtext("xmax",iw)))); y2=min(ih,int(float(bb.findtext("ymax",ih))))
            if x2>x1 and y2>y1:
                records_minhkb.append({"img_path":ip,"class":cls,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"source":"minhkb"})
                class_counts_m[cls] += 1
    except: pass

print(f"  MinhNKB records: {len(records_minhkb)}")
print(f"  MinhNKB class dist: {class_counts_m}")

print("\n[1b] Parsing Jomarkow dataset (YOLO format)...")
records_jomarkow = []
class_counts_j = {c:0 for c in JOMARKOW_CLASS_MAP.values()}
lbl_files = sorted(glob.glob(os.path.join(JOMARKOW_LBL_DIR,"*.txt")))

for lf in lbl_files:
    img_base = os.path.splitext(os.path.basename(lf))[0]
    ip = os.path.join(JOMARKOW_IMG_DIR, img_base+".png")
    if not os.path.exists(ip): ip = os.path.join(JOMARKOW_IMG_DIR, img_base+".jpg")
    if not os.path.exists(ip): continue
    try:
        img_tmp = cv2.imread(ip)
        if img_tmp is None: continue
        ih, iw = img_tmp.shape[:2]
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls_id = int(parts[0])
                if cls_id not in JOMARKOW_CLASS_MAP: continue
                cls = JOMARKOW_CLASS_MAP[cls_id]
                if class_counts_j[cls] >= MAX_PER_CLASS_JOMARKOW: continue
                cx,cy,bw,bh = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
                x1=max(0,int((cx-bw/2)*iw)); y1=max(0,int((cy-bh/2)*ih))
                x2=min(iw,int((cx+bw/2)*iw)); y2=min(ih,int((cy+bh/2)*ih))
                if x2>x1+8 and y2>y1+8:
                    records_jomarkow.append({"img_path":ip,"class":cls,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"source":"jomarkow"})
                    class_counts_j[cls] += 1
    except: pass

print(f"  Jomarkow records: {len(records_jomarkow)}")
print(f"  Jomarkow class dist: {class_counts_j}")

all_records = records_minhkb + records_jomarkow
print(f"\n  COMBINED total records: {len(all_records)}")

# ─────────────────────────────────────────────────────────────────
# SECTION 2: EXTRACT FEATURES + AUGMENT
# ─────────────────────────────────────────────────────────────────
print("\n[2/5] Extracting features with augmentation...")
random.shuffle(all_records)

features_ml, labels_ml, labels_binary = [], [], []
crops_cnn, labels_cnn = [], []
MAX_TOTAL_PER_CLASS = 700  # after augmentation, cap total

class_feat_counts = {c:0 for c in ALL_CLASSES}

for rec in all_records:
    cls = rec['class']
    if class_feat_counts[cls] >= MAX_TOTAL_PER_CLASS: continue
    try:
        img = cv2.imread(rec['img_path'])
        if img is None: continue
        crop = img[rec['y1']:rec['y2'], rec['x1']:rec['x2']]
        if crop.size==0 or crop.shape[0]<8 or crop.shape[1]<8: continue

        # Apply augmentation for under-represented classes
        n_needed = MAX_TOTAL_PER_CLASS - class_feat_counts[cls]
        variants = augment_crop(crop)[:min(len(augment_crop(crop)), max(1, n_needed))]

        for var in variants:
            if class_feat_counts[cls] >= MAX_TOTAL_PER_CLASS: break
            # ML features
            feat = extract_features(var)
            features_ml.append(feat)
            labels_ml.append(cls)
            labels_binary.append(BINARY_MAP[cls])
            class_feat_counts[cls] += 1
            # CNN crop (resized, normalized)
            crop_rgb = cv2.cvtColor(cv2.resize(var, CROP_SIZE), cv2.COLOR_BGR2RGB)
            crops_cnn.append(crop_rgb)
            labels_cnn.append(cls)
    except: pass

X = np.array(features_ml); y_m = np.array(labels_ml); y_b = np.array(labels_binary)
print(f"  Feature matrix: {X.shape}")
print(f"  Final class balance: {dict(zip(*np.unique(y_m, return_counts=True)))}")

# ─────────────────────────────────────────────────────────────────
# SECTION 3: TRAIN ML MODELS
# ─────────────────────────────────────────────────────────────────
print("\n[3/5] Training ML models (SVM, RF, GBM)...")

le_m = LabelEncoder(); y_m_enc = le_m.fit_transform(y_m)
le_b = LabelEncoder(); y_b_enc = le_b.fit_transform(y_b)
Xtr,Xte,yt,yv = train_test_split(X,y_m_enc,test_size=0.2,random_state=42,stratify=y_m_enc)
Xtr_b,Xte_b,yt_b,yv_b = train_test_split(X,y_b_enc,test_size=0.2,random_state=42,stratify=y_b_enc)
print(f"  Train={len(Xtr)}, Test={len(Xte)} | Classes: {list(le_m.classes_)}")

results = {}
def run(name, pipe, Xtr, Xte, ytr, yte, le, task, save_key):
    t0=time.time(); pipe.fit(Xtr,ytr); tt=time.time()-t0
    yp=pipe.predict(Xte)
    rpt=classification_report(yte,yp,target_names=le.classes_,output_dict=True)
    cm=confusion_matrix(yte,yp); acc=rpt['accuracy']
    print(f"  [{name:30s}] acc={acc:.4f}  train={tt:.1f}s")
    r = {"name":name,"task":task,"accuracy":acc,"report":rpt,"confusion":cm,
         "model":pipe,"train_time":tt,"y_pred":yp,"y_test":yte,"le":le}
    results[save_key] = r
    joblib.dump(pipe, os.path.join(OUT_DIR,f"v2_model_{save_key}.pkl"))
    return r

run("SVM RBF (multi)", Pipeline([
    ('sc',StandardScaler()),('pca',PCA(n_components=150,random_state=42)),
    ('svm',SVC(kernel='rbf',C=15,gamma='scale',probability=True,random_state=42))
]), Xtr,Xte,yt,yv,le_m,"multi","svm_multi")

run("SVM RBF (binary)", Pipeline([
    ('sc',StandardScaler()),('pca',PCA(n_components=100,random_state=42)),
    ('svm',SVC(kernel='rbf',C=10,gamma='scale',probability=True,random_state=42))
]), Xtr_b,Xte_b,yt_b,yv_b,le_b,"binary","svm_binary")

run("Random Forest (multi)", Pipeline([
    ('sc',StandardScaler()),
    ('rf',RandomForestClassifier(n_estimators=400,max_depth=20,min_samples_split=3,
                                  n_jobs=-1,random_state=42))
]), Xtr,Xte,yt,yv,le_m,"multi","rf_multi")

run("Random Forest (binary)", Pipeline([
    ('sc',StandardScaler()),
    ('rf',RandomForestClassifier(n_estimators=400,max_depth=16,n_jobs=-1,random_state=42))
]), Xtr_b,Xte_b,yt_b,yv_b,le_b,"binary","rf_binary")

run("GradBoost (multi)", Pipeline([
    ('sc',StandardScaler()),('pca',PCA(n_components=100,random_state=42)),
    ('gbm',HistGradientBoostingClassifier(max_iter=200,max_depth=8,
                                          learning_rate=0.05,random_state=42))
]), Xtr,Xte,yt,yv,le_m,"multi","gbm_multi")

run("GradBoost (binary)", Pipeline([
    ('sc',StandardScaler()),('pca',PCA(n_components=80,random_state=42)),
    ('gbm',HistGradientBoostingClassifier(max_iter=200,max_depth=6,
                                          learning_rate=0.05,random_state=42))
]), Xtr_b,Xte_b,yt_b,yv_b,le_b,"binary","gbm_binary")

# ─────────────────────────────────────────────────────────────────
# SECTION 4: CNN (25 epochs, pre-cached)
# ─────────────────────────────────────────────────────────────────
print("\n[4/5] Training CNN (25 epochs)...")

class SmallCNN(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)),
        )
        self.fc=nn.Sequential(
            nn.Dropout(0.4),nn.Linear(256*4,512),nn.ReLU(),
            nn.Dropout(0.3),nn.Linear(512,nc)
        )
    def forward(self,x): return self.fc(self.net(x).flatten(1))

# Encode labels
le_cnn = LabelEncoder()
y_cnn_enc = le_cnn.fit_transform(labels_cnn)
# Cache all crops as tensor
X_arr = np.stack(crops_cnn).astype(np.float32)/255.0
mean=np.array([0.485,0.456,0.406],dtype=np.float32); std=np.array([0.229,0.224,0.225],dtype=np.float32)
X_arr=(X_arr-mean)/std
X_t = torch.tensor(X_arr.transpose(0,3,1,2))
y_t = torch.tensor(y_cnn_enc,dtype=torch.long)

tr_idx,te_idx = train_test_split(range(len(y_cnn_enc)),test_size=0.2,random_state=42,stratify=y_cnn_enc)
tr_ds=TensorDataset(X_t[tr_idx],y_t[tr_idx]); te_ds=TensorDataset(X_t[te_idx],y_t[te_idx])
tr_dl=DataLoader(tr_ds,batch_size=64,shuffle=True); te_dl=DataLoader(te_ds,batch_size=64)
print(f"  CNN Train={len(tr_idx)}, Test={len(te_idx)} | Classes: {list(le_cnn.classes_)}")

device=torch.device('cpu')
model=SmallCNN(len(le_cnn.classes_)).to(device)
opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=25)
crit=nn.CrossEntropyLoss()

EPOCHS=25
tr_accs,va_accs,tr_losses,va_losses=[],[],[],[]
best_acc,best_state=0,None
t0=time.time()

for ep in range(EPOCHS):
    model.train(); tl,tc,tt=0,0,0
    for imgs,lbls in tr_dl:
        opt.zero_grad(); out=model(imgs); loss=crit(out,lbls)
        loss.backward(); opt.step()
        tl+=loss.item()*len(lbls); tc+=(out.argmax(1)==lbls).sum().item(); tt+=len(lbls)
    model.eval(); vl,vc,vt=0,0,0
    with torch.no_grad():
        for imgs,lbls in te_dl:
            out=model(imgs); vl+=crit(out,lbls).item()*len(lbls)
            vc+=(out.argmax(1)==lbls).sum().item(); vt+=len(lbls)
    ta,va=tc/tt,vc/vt
    tr_accs.append(ta); va_accs.append(va)
    tr_losses.append(tl/tt); va_losses.append(vl/vt)
    sched.step()
    if va>best_acc: best_acc=va; best_state={k:v.clone() for k,v in model.state_dict().items()}
    print(f"    Ep {ep+1:>2}/{EPOCHS}  tr={ta:.3f}  val={va:.3f}  lr={sched.get_last_lr()[0]:.5f}")

cnn_time=time.time()-t0
print(f"  CNN done {cnn_time:.1f}s | best_val={best_acc:.3f}")
torch.save(best_state, os.path.join(OUT_DIR,"v2_model_cnn_multi.pth"))

model.load_state_dict(best_state); model.eval()
all_p,all_l=[],[]
with torch.no_grad():
    for imgs,lbls in te_dl:
        all_p.extend(model(imgs).argmax(1).numpy()); all_l.extend(lbls.numpy())

cnn_rpt=classification_report(all_l,all_p,target_names=le_cnn.classes_,output_dict=True)
cnn_cm=confusion_matrix(all_l,all_p)
results['cnn_multi']={
    "name":"CNN v2 (multi)","task":"multi",
    "accuracy":cnn_rpt['accuracy'],"report":cnn_rpt,"confusion":cnn_cm,
    "model":model,"train_time":cnn_time,"y_pred":np.array(all_p),"y_test":np.array(all_l),
    "le":le_cnn,"tr_accs":tr_accs,"va_accs":va_accs,"tr_losses":tr_losses,"va_losses":va_losses
}
print(f"\n  CNN multi-class report:")
print(classification_report(all_l,all_p,target_names=le_cnn.classes_))

# ─────────────────────────────────────────────────────────────────
# SECTION 5: TWO-STAGE CCTV DETECTION PIPELINE
# ─────────────────────────────────────────────────────────────────
print("\n[5/5] Two-stage CCTV detection pipeline...")

# Stage 1: HOG person detector
hog_person = cv2.HOGDescriptor()
hog_person.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people(img_bgr, scale=1.05, win_stride=8, padding=8, min_size=(48,96)):
    """Use OpenCV HOG to detect people bounding boxes"""
    h,w=img_bgr.shape[:2]
    # Resize for better detection on CCTV images
    scale_factor = 1.0
    if w > 1280:
        scale_factor = 1280/w
        img_bgr = cv2.resize(img_bgr, (int(w*scale_factor), int(h*scale_factor)))
    found_rects, weights = hog_person.detectMultiScale(
        img_bgr,
        winStride=(win_stride,win_stride),
        padding=(padding,padding),
        scale=scale,
        useMeanshiftGrouping=False,
    )
    boxes = []
    if len(found_rects)>0:
        for (bx,by,bw,bh),w_ in zip(found_rects,weights):
            if float(w_) > 0.3:  # confidence threshold
                # Scale back to original
                x1=int(bx/scale_factor); y1=int(by/scale_factor)
                x2=int((bx+bw)/scale_factor); y2=int((by+bh)/scale_factor)
                boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'confidence':float(w_)})
    return boxes

def cnn_classify_crop(crop_bgr, cnn_model, le):
    """Classify a single crop using CNN"""
    crop_rgb = cv2.cvtColor(cv2.resize(crop_bgr, CROP_SIZE), cv2.COLOR_BGR2RGB)
    x = torch.tensor(crop_rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    mn=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    sd=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    x=(x-mn)/sd
    with torch.no_grad():
        out=cnn_model(x); probs=torch.softmax(out,1).squeeze().numpy()
    return le.classes_[probs.argmax()], probs.max()

def svm_classify_crop(crop_bgr, svm_model, le):
    """Classify a single crop using SVM"""
    feat = extract_features(crop_bgr).reshape(1,-1)
    cls_enc = svm_model.predict(feat)[0]
    proba = svm_model.predict_proba(feat)[0].max()
    return le.inverse_transform([cls_enc])[0], proba

# Color map for PPE classes
CLASS_COLORS_BGR = {
    'helmet':      (39,174,96),    # green
    'safety_vest': (192,57,43),    # blue
    'full_ppe':    (142,68,173),   # purple
    'partial_ppe': (243,156,18),   # orange
    'no_ppe':      (231,76,60),    # red
}
CLASS_COLORS_HEX = {k:'#{:02x}{:02x}{:02x}'.format(v[2],v[1],v[0])
                    for k,v in CLASS_COLORS_BGR.items()}

print("  Running two-stage detection on CCTV validation images...")
val_imgs = sorted(
    glob.glob(os.path.join(VAL_DIR,"*.jpg")) +
    glob.glob(os.path.join(VAL_DIR,"*.JPG")) +
    glob.glob(os.path.join(VAL_DIR,"*.png"))
)

# Load best SVM
svm_model = results['svm_multi']['model']

fig, axes = plt.subplots(3, 4, figsize=(22, 16))
axes = axes.flatten()
fig.suptitle("Two-Stage PPE Detection: Person Detection → PPE Classification\n"
             "(Stage 1: OpenCV HOG Person Detector | Stage 2: CNN Classifier)",
             fontsize=13, fontweight='bold')
fig.patch.set_facecolor('#111111')

summary_rows = []

for idx, img_path in enumerate(val_imgs[:12]):
    img = cv2.imread(img_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = axes[idx]
    ax.imshow(img_rgb)
    ax.set_facecolor('#111111')

    h_orig,w_orig = img.shape[:2]
    is_scene = h_orig > 200 and w_orig > 200

    if is_scene:
        # ── STAGE 1: Detect people ──────────────────────────────
        people_boxes = detect_people(img.copy(), scale=1.05, win_stride=8, padding=8)

        # ── STAGE 2: Classify each person ───────────────────────
        detections = []
        for box in people_boxes:
            x1,y1,x2,y2 = box['x1'],box['y1'],box['x2'],box['y2']
            # Pad crop slightly
            pad=10
            px1=max(0,x1-pad); py1=max(0,y1-pad)
            px2=min(w_orig,x2+pad); py2=min(h_orig,y2+pad)
            person_crop = img[py1:py2,px1:px2]
            if person_crop.size==0 or person_crop.shape[0]<20 or person_crop.shape[1]<20:
                continue
            # Classify using CNN
            cls,conf = cnn_classify_crop(person_crop, model, le_cnn)
            detections.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'class':cls,'conf':float(conf)})

        # Draw detection boxes
        for det in detections:
            hex_col = CLASS_COLORS_HEX.get(det['class'],'gray')
            rect = patches.Rectangle(
                (det['x1'],det['y1']), det['x2']-det['x1'], det['y2']-det['y1'],
                lw=2.5, edgecolor=hex_col, facecolor='none', alpha=0.9
            )
            ax.add_patch(rect)
            ax.text(det['x1'],det['y1']-4,
                    f"{'P'} → {det['class'][:8]} {det['conf']:.2f}",
                    color='white', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=hex_col, alpha=0.85))

        n_p=len(people_boxes)
        n_d=len(detections)
        n_violation = sum(1 for d in detections if d['class']=='no_ppe')
        col_title = 'red' if n_violation>0 else 'lime'
        ax.set_title(f"{os.path.basename(img_path)}\n"
                     f"People: {n_p} detected | Classified: {n_d} | ⚠ Violations: {n_violation}",
                     fontsize=8, color=col_title, pad=4)
        summary_rows.append({'image':os.path.basename(img_path),
                              'people_detected':n_p,'classified':n_d,
                              'violations':n_violation,'stage':'two_stage'})
    else:
        # Direct crop classification
        cls,conf = cnn_classify_crop(img, model, le_cnn)
        hex_col = CLASS_COLORS_HEX.get(cls,'gray')
        rect = patches.Rectangle((2,2),w_orig-4,h_orig-4,lw=3,
                                  edgecolor=hex_col,facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"{os.path.basename(img_path)}\n{cls} ({conf:.2f})",
                     fontsize=8,color=hex_col)
        summary_rows.append({'image':os.path.basename(img_path),
                              'people_detected':1,'classified':1,
                              'violations':1 if cls=='no_ppe' else 0,'stage':'direct'})
    ax.axis('off')

for j in range(len(val_imgs), 12):
    axes[j].axis('off')
    axes[j].set_facecolor('#111111')

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0],color=v,lw=3,label=k) for k,v in CLASS_COLORS_HEX.items()]
legend_elements.append(Line2D([0],[0],color='white',lw=0,
    label='Boxes = Person detections (Stage 1 → Stage 2)'))
fig.legend(handles=legend_elements,loc='lower center',ncol=6,fontsize=9,
           bbox_to_anchor=(0.5,-0.01),facecolor='#222222',labelcolor='white',
           edgecolor='gray')

plt.tight_layout(pad=2)
plt.savefig(os.path.join(OUT_DIR,"v2_cctv_two_stage.png"),dpi=150,bbox_inches='tight',
            facecolor='#111111')
plt.close()
print("  Saved v2_cctv_two_stage.png")

# ─────────────────────────────────────────────────────────────────
# SECTION 6: COMPARISON PLOTS + SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\nGenerating comparison plots...")

# CNN training curves
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
ep_range=range(1,EPOCHS+1)
ax1.plot(ep_range,results['cnn_multi']['tr_losses'],'r-',lw=2,label='Train Loss')
ax1.plot(ep_range,results['cnn_multi']['va_losses'],'b-',lw=2,label='Val Loss')
ax1.fill_between(ep_range,results['cnn_multi']['tr_losses'],alpha=0.15,color='red')
ax1.fill_between(ep_range,results['cnn_multi']['va_losses'],alpha=0.15,color='blue')
ax1.set_title("CNN v2 Training Loss (25 epochs)",fontsize=12,fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(ep_range,results['cnn_multi']['tr_accs'],'r-',lw=2,label='Train Acc')
ax2.plot(ep_range,results['cnn_multi']['va_accs'],'b-',lw=2,label='Val Acc')
ax2.axhline(best_acc,color='green',lw=1.5,linestyle='--',label=f'Best={best_acc:.3f}')
ax2.fill_between(ep_range,results['cnn_multi']['tr_accs'],alpha=0.15,color='red')
ax2.fill_between(ep_range,results['cnn_multi']['va_accs'],alpha=0.15,color='blue')
ax2.set_title("CNN v2 Accuracy (25 epochs)",fontsize=12,fontweight='bold')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
ax2.set_ylim(0,1.05)
plt.suptitle("CNN v2 Training History — Combined Dataset",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"v2_cnn_training.png"),dpi=150,bbox_inches='tight')
plt.close()

# Full model comparison
all_multi = {k:v for k,v in results.items() if v['task']=='multi'}
all_binary = {k:v for k,v in results.items() if v['task']=='binary'}

fig,axes=plt.subplots(1,2,figsize=(16,6))
for ax,(res_d,title) in zip(axes,[(all_multi,"Multi-class Accuracy"),(all_binary,"Binary Accuracy")]):
    names=[v['name'] for v in res_d.values()]
    accs=[v['accuracy'] for v in res_d.values()]
    cols=['#e74c3c','#3498db','#2ecc71','#f39c12'][:len(names)]
    bars=ax.bar(range(len(names)),accs,color=cols,edgecolor='black',width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names,rotation=20,fontsize=9)
    ax.set_ylim(0,1.1); ax.set_title(title,fontsize=13,fontweight='bold'); ax.set_ylabel("Accuracy")
    ax.axhline(0.8,color='red',lw=1,linestyle='--',alpha=0.5,label='80% line')
    for bar,a in zip(bars,accs):
        ax.text(bar.get_x()+bar.get_width()/2,a+0.01,f'{a:.3f}',
                ha='center',va='bottom',fontweight='bold',fontsize=10)
plt.suptitle("Model Comparison v2 — Combined Dataset (MinhNKB + Jomarkow)\n"
             "with Augmentation + Longer Training",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"v2_model_comparison.png"),dpi=150,bbox_inches='tight')
plt.close()

# Confusion matrices (multi)
fig,axes=plt.subplots(1,4,figsize=(24,5))
for ax,(k,res) in zip(axes,all_multi.items()):
    ConfusionMatrixDisplay(res['confusion'],display_labels=res['le'].classes_).plot(
        ax=ax,cmap='Blues',colorbar=False,values_format='d')
    ax.set_title(f"{res['name']}\nacc={res['accuracy']:.3f}",fontsize=9)
    ax.tick_params(axis='x',rotation=35)
plt.suptitle("Confusion Matrices v2 — Multi-class PPE Detection",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"v2_confusion_matrices.png"),dpi=150,bbox_inches='tight')
plt.close()

# F1 heatmap
cls_names=list(le_m.classes_)
f1d={}
for k,res in all_multi.items():
    f1d[res['name']]=[res['report'].get(c,{}).get('f1-score',0) for c in cls_names]
fig,ax=plt.subplots(figsize=(12,4))
sns.heatmap(pd.DataFrame(f1d,index=cls_names),annot=True,fmt='.3f',
            cmap='YlOrRd',ax=ax,linewidths=0.5,cbar_kws={'label':'F1'})
ax.set_title("Per-class F1 Score — v2 Models on Combined Dataset",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"v2_f1_heatmap.png"),dpi=150,bbox_inches='tight')
plt.close()

# ROC curves
fig,ax=plt.subplots(figsize=(8,6))
for (k,res),col in zip(all_binary.items(),['#e74c3c','#3498db','#2ecc71']):
    if hasattr(res['model'],'predict_proba'):
        ys=res['model'].predict_proba(Xte_b)[:,1]
        fpr,tpr,_=roc_curve(yv_b,ys); ra=auc(fpr,tpr)
        ax.plot(fpr,tpr,color=col,lw=2,label=f"{res['name']} (AUC={ra:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves v2 — Binary PPE Classification",fontsize=13,fontweight='bold')
ax.legend(loc='lower right'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"v2_roc_curves.png"),dpi=150,bbox_inches='tight')
plt.close()

# ─── Summary CSV ───────────────────────────────────────────────
rows=[]
for k,res in results.items():
    rpt=res['report']
    rows.append({'Model':res['name'],'Task':res['task'],
                 'Accuracy':round(res['accuracy'],4),
                 'Macro F1':round(rpt.get('macro avg',{}).get('f1-score',0),4),
                 'Weighted F1':round(rpt.get('weighted avg',{}).get('f1-score',0),4),
                 'Train Time(s)':round(res['train_time'],1)})
summary=pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR,"v2_model_summary.csv"),index=False)
pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR,"v2_cctv_summary.csv"),index=False)

print("\n"+"="*65)
print("RESULTS SUMMARY v2 — COMBINED DATASET")
print("="*65)
print(summary.to_string(index=False))
print("="*65)

# Dataset size comparison
print(f"\n  Dataset comparison:")
print(f"  v1: MinhNKB only — 2000 crops (400/class)")
print(f"  v2: MinhNKB+Jomarkow+Augmentation — {len(X)} crops")
print(f"  Training set growth: {len(X)/2000:.1f}x")

print("\nAll v2 outputs saved!")
for f in sorted(os.listdir(OUT_DIR)):
    if 'v2' in f:
        sz=os.path.getsize(os.path.join(OUT_DIR,f))
        print(f"  {f:48s} {sz/1024:6.1f} KB")
