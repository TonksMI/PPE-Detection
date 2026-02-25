"""
PPE v2 CNN Training — Combined Dataset, Optimized
Caches all crops in RAM first, then fast training
"""
import os, glob, random, time, warnings
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42); torch.manual_seed(42)

MINHNKB_IMG_DIR = "/sessions/sleepy-epic-pascal/datasets/helmet-safety-vest-detection-master/train-images-data"
MINHNKB_ANN_DIR = "/sessions/sleepy-epic-pascal/datasets/helmet-safety-vest-detection-master/train-images-annotations-new"
JOMARKOW_IMG_DIR = "/sessions/sleepy-epic-pascal/datasets/jomarkow/images"
JOMARKOW_LBL_DIR = "/sessions/sleepy-epic-pascal/datasets/jomarkow/labels"
OUT_DIR = "/sessions/sleepy-epic-pascal/mnt/Computer Vision"

MINHNKB_MAP = {"helmet":"helmet","safety vest":"safety_vest",
    "person with full safety":"full_ppe","person with partial safety":"partial_ppe",
    "person without safety":"no_ppe"}
JOMARKOW_MAP = {0:"helmet", 1:"no_ppe"}

print("="*55); print("PPE CNN v2 — Combined Dataset Training"); print("="*55)

# ── Batch-load all crops ───────────────────────────────────────
print("\nLoading crops from BOTH datasets...")
MAX = 500   # per class per dataset
crops, labels = [], []
cc = {}

t0=time.time()
def add_crop(img_bgr, cls, augment=True):
    global crops, labels, cc
    if cc.get(cls,0) >= MAX: return
    crop_rgb = cv2.cvtColor(cv2.resize(img_bgr,(64,64)), cv2.COLOR_BGR2RGB)
    crops.append(crop_rgb); labels.append(cls); cc[cls]=cc.get(cls,0)+1
    if augment and cc.get(cls,0) < MAX:
        # Flip
        crops.append(cv2.cvtColor(cv2.flip(cv2.resize(img_bgr,(64,64)),1),cv2.COLOR_BGR2RGB))
        labels.append(cls); cc[cls]+=1
    if cc.get(cls,0) < MAX and augment:
        # Brightness
        bright = np.clip(cv2.resize(img_bgr,(64,64)).astype(np.float32)*1.3,0,255).astype(np.uint8)
        crops.append(cv2.cvtColor(bright,cv2.COLOR_BGR2RGB)); labels.append(cls); cc[cls]+=1

# ── MinhNKB ────────────────────────────────────────────────────
xml_files = sorted(glob.glob(os.path.join(MINHNKB_ANN_DIR,"*.xml")))
random.shuffle(xml_files)
for xf in xml_files:
    try:
        root=ET.parse(xf).getroot()
        fname=root.findtext("filename")
        ip=os.path.join(MINHNKB_IMG_DIR,fname) if fname else None
        if not ip or not os.path.exists(ip):
            stem=os.path.splitext(os.path.basename(xf))[0]
            for e in [".jpg",".jpeg",".png"]:
                c=os.path.join(MINHNKB_IMG_DIR,stem+e)
                if os.path.exists(c): ip=c; break
        if not ip or not os.path.exists(ip): continue
        img=cv2.imread(ip)
        if img is None: continue
        sz=root.find("size"); iw,ih=int(sz.findtext("width",0)),int(sz.findtext("height",0))
        for obj in root.findall("object"):
            raw=obj.findtext("name","").strip().lower()
            if raw not in MINHNKB_MAP: continue
            cls=MINHNKB_MAP[raw]
            if cc.get(cls,0)>=MAX: continue
            bb=obj.find("bndbox")
            x1=max(0,int(float(bb.findtext("xmin",0)))); y1=max(0,int(float(bb.findtext("ymin",0))))
            x2=min(iw,int(float(bb.findtext("xmax",iw)))); y2=min(ih,int(float(bb.findtext("ymax",ih))))
            if x2>x1+8 and y2>y1+8:
                add_crop(img[y1:y2,x1:x2], cls)
    except: pass

print(f"  After MinhNKB: {len(crops)} crops | {cc}")

# ── Jomarkow ───────────────────────────────────────────────────
# Reset per-dataset caps with separate counter
cc_j = {}
MAX_J = 400
lbl_files=sorted(glob.glob(os.path.join(JOMARKOW_LBL_DIR,"*.txt")))
random.shuffle(lbl_files)
for lf in lbl_files:
    base=os.path.splitext(os.path.basename(lf))[0]
    ip=os.path.join(JOMARKOW_IMG_DIR,base+".png")
    if not os.path.exists(ip): ip=os.path.join(JOMARKOW_IMG_DIR,base+".jpg")
    if not os.path.exists(ip): continue
    try:
        img=cv2.imread(ip)
        if img is None: continue
        ih,iw=img.shape[:2]
        for line in open(lf):
            parts=line.strip().split()
            if len(parts)<5: continue
            cid=int(parts[0])
            if cid not in JOMARKOW_MAP: continue
            cls=JOMARKOW_MAP[cid]
            if cc_j.get(cls,0)>=MAX_J: continue
            cx,cy,bw,bh=float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
            x1=max(0,int((cx-bw/2)*iw)); y1=max(0,int((cy-bh/2)*ih))
            x2=min(iw,int((cx+bw/2)*iw)); y2=min(ih,int((cy+bh/2)*ih))
            if x2>x1+8 and y2>y1+8:
                crop=img[y1:y2,x1:x2]
                crop_rgb=cv2.cvtColor(cv2.resize(crop,(64,64)),cv2.COLOR_BGR2RGB)
                crops.append(crop_rgb); labels.append(cls)
                cc_j[cls]=cc_j.get(cls,0)+1
    except: pass

print(f"  After Jomarkow: {len(crops)} crops | jomarkow={cc_j}")
print(f"  Final class dist: {dict(sorted({l:labels.count(l) for l in set(labels)}.items()))}")
print(f"  Load time: {time.time()-t0:.1f}s")

# ── Prepare tensors ────────────────────────────────────────────
le=LabelEncoder(); y_enc=le.fit_transform(labels)
X_arr=np.stack(crops).astype(np.float32)/255.0
mean=np.array([0.485,0.456,0.406],dtype=np.float32)
std=np.array([0.229,0.224,0.225],dtype=np.float32)
X_arr=(X_arr-mean)/std
X_t=torch.tensor(X_arr.transpose(0,3,1,2)); y_t=torch.tensor(y_enc,dtype=torch.long)
print(f"\nTensor shape: {X_t.shape}")

tr_idx,te_idx=train_test_split(range(len(y_enc)),test_size=0.2,random_state=42,stratify=y_enc)
tr_dl=DataLoader(TensorDataset(X_t[tr_idx],y_t[tr_idx]),batch_size=64,shuffle=True)
te_dl=DataLoader(TensorDataset(X_t[te_idx],y_t[te_idx]),batch_size=64)
print(f"Train={len(tr_idx)}, Test={len(te_idx)} | Classes={list(le.classes_)}")

# ── Model ──────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d((2,2)),
        )
        self.fc=nn.Sequential(nn.Dropout(0.4),nn.Linear(256*4,512),nn.ReLU(),
                              nn.Dropout(0.3),nn.Linear(512,nc))
    def forward(self,x): return self.fc(self.net(x).flatten(1))

model=SmallCNN(len(le.classes_))
opt=optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
crit=nn.CrossEntropyLoss()

EPOCHS=20
tr_accs,va_accs,tr_losses,va_losses=[],[],[],[]
best_acc,best_state=0,None
t0=time.time()
print(f"\nTraining {EPOCHS} epochs on CPU...")

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
    print(f"  Ep {ep+1:>2}/{EPOCHS}  tr={ta:.3f}  val={va:.3f}  lr={sched.get_last_lr()[0]:.6f}")

print(f"\nDone {time.time()-t0:.1f}s | best_val={best_acc:.3f}")
torch.save(best_state, os.path.join(OUT_DIR,"v2_model_cnn_multi.pth"))

model.load_state_dict(best_state); model.eval()
all_p,all_l=[],[]
with torch.no_grad():
    for imgs,lbls in te_dl:
        all_p.extend(model(imgs).argmax(1).numpy()); all_l.extend(lbls.numpy())

rpt=classification_report(all_l,all_p,target_names=le.classes_,output_dict=True)
cm=confusion_matrix(all_l,all_p)
cnn_time=time.time()-t0
print("\n"+classification_report(all_l,all_p,target_names=le.classes_))

# ── Plots ──────────────────────────────────────────────────────
ep_r=range(1,EPOCHS+1)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
ax1.plot(ep_r,tr_losses,'r-',lw=2,label='Train'); ax1.plot(ep_r,va_losses,'b-',lw=2,label='Val')
ax1.fill_between(ep_r,tr_losses,alpha=0.1,color='red'); ax1.fill_between(ep_r,va_losses,alpha=0.1,color='blue')
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(ep_r,tr_accs,'r-',lw=2,label='Train'); ax2.plot(ep_r,va_accs,'b-',lw=2,label='Val')
ax2.axhline(best_acc,c='green',lw=1.5,ls='--',label=f'Best={best_acc:.3f}')
ax2.fill_between(ep_r,tr_accs,alpha=0.1,color='red'); ax2.fill_between(ep_r,va_accs,alpha=0.1,color='blue')
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0,1.05)
plt.suptitle(f"CNN v2 — Combined Dataset ({len(crops)} crops, {EPOCHS} epochs)",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"v2_cnn_training.png"),dpi=150,bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay(cm,display_labels=le.classes_).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
ax.set_title(f"CNN v2 Confusion Matrix\nacc={rpt['accuracy']:.3f}"); ax.tick_params(axis='x',rotation=35)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"v2_cnn_confusion.png"),dpi=150,bbox_inches='tight'); plt.close()

# Append to ML summary
cnn_row={'Model':'CNN v2 (multi-class)','Task':'multi','Accuracy':round(rpt['accuracy'],4),
         'Macro F1':round(rpt['macro avg']['f1-score'],4),
         'Weighted F1':round(rpt['weighted avg']['f1-score'],4),
         'Train Time(s)':round(cnn_time,1)}
try:
    ml_summary=pd.read_csv(os.path.join(OUT_DIR,"v2_model_summary_ml.csv"))
    full=pd.concat([ml_summary,pd.DataFrame([cnn_row])],ignore_index=True)
except:
    full=pd.DataFrame([cnn_row])
full.to_csv(os.path.join(OUT_DIR,"v2_model_summary.csv"),index=False)

# Save label encoder + CNN state for two-stage pipeline
import pickle
with open("/sessions/sleepy-epic-pascal/v2_le_cnn.pkl","wb") as f:
    pickle.dump(le,f)

print("\nCNN v2 summary:")
print(full.to_string(index=False))
