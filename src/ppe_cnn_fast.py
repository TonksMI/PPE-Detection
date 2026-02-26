"""
PPE CNN — Pre-cache crops in memory, then train fast
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

# Auto-detect paths (works on both Linux and Windows)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # PPE-Detection/
BASE = os.path.dirname(PROJECT_DIR)  # D:\Claude or /sessions/...

DATASET_DIR = os.path.join(BASE, "datasets/helmet-safety-vest-detection-master")
IMG_DIR  = os.path.join(DATASET_DIR, "train-images-data")
ANN_DIR  = os.path.join(DATASET_DIR, "train-images-annotations-new")
OUT_DIR  = os.path.join(PROJECT_DIR, "results/models")

RAW_CLASS_MAP = {"helmet":"helmet","safety vest":"safety_vest",
    "person with full safety":"full_ppe","person with partial safety":"partial_ppe",
    "person without safety":"no_ppe"}

print("Loading & caching crops...")
records,crops_cached,labels=[],[],[]
MAX=300
cc={c:0 for c in RAW_CLASS_MAP.values()}

for xf in sorted(glob.glob(os.path.join(ANN_DIR,"*.xml"))):
    try:
        root=ET.parse(xf).getroot()
        fname=root.findtext("filename")
        ip=os.path.join(IMG_DIR,fname) if fname else None
        if not ip or not os.path.exists(ip):
            stem=os.path.splitext(os.path.basename(xf))[0]
            for e in [".jpg",".jpeg",".png"]:
                c=os.path.join(IMG_DIR,stem+e)
                if os.path.exists(c): ip=c; break
        if not ip or not os.path.exists(ip): continue
        img=cv2.imread(ip)
        if img is None: continue
        sz=root.find("size"); iw,ih=int(sz.findtext("width",0)),int(sz.findtext("height",0))
        for obj in root.findall("object"):
            raw=obj.findtext("name","").strip().lower()
            if raw not in RAW_CLASS_MAP: continue
            cls=RAW_CLASS_MAP[raw]
            if cc[cls]>=MAX: continue
            bb=obj.find("bndbox")
            x1=max(0,int(float(bb.findtext("xmin",0)))); y1=max(0,int(float(bb.findtext("ymin",0))))
            x2=min(iw,int(float(bb.findtext("xmax",iw)))); y2=min(ih,int(float(bb.findtext("ymax",ih))))
            if x2<=x1 or y2<=y1: continue
            crop=img[y1:y2,x1:x2]
            if crop.size==0 or crop.shape[0]<8 or crop.shape[1]<8: continue
            crop_rgb=cv2.cvtColor(cv2.resize(crop,(64,64)),cv2.COLOR_BGR2RGB)
            crops_cached.append(crop_rgb); labels.append(cls); cc[cls]+=1
    except: pass

print(f"Cached {len(crops_cached)} crops: {dict(zip(*np.unique(labels,return_counts=True)))}")

le=LabelEncoder(); y_enc=le.fit_transform(labels)
X_arr=np.stack(crops_cached).astype(np.float32)/255.0
mean=np.array([0.485,0.456,0.406],dtype=np.float32); std=np.array([0.229,0.224,0.225],dtype=np.float32)
X_arr=(X_arr-mean)/std
X_t=torch.tensor(X_arr.transpose(0,3,1,2)); y_t=torch.tensor(y_enc,dtype=torch.long)

tr_idx,te_idx=train_test_split(range(len(y_enc)),test_size=0.2,random_state=42,stratify=y_enc)
tr_ds=TensorDataset(X_t[tr_idx],y_t[tr_idx]); te_ds=TensorDataset(X_t[te_idx],y_t[te_idx])
tr_dl=DataLoader(tr_ds,batch_size=64,shuffle=True); te_dl=DataLoader(te_ds,batch_size=64)

print(f"Train={len(tr_idx)} | Test={len(te_idx)} | Classes={list(le.classes_)}")

class SmallCNN(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2,2)),
        )
        self.fc=nn.Sequential(nn.Dropout(0.4),nn.Linear(128*4,128),nn.ReLU(),nn.Linear(128,nc))
    def forward(self,x): return self.fc(self.net(x).flatten(1))

device=torch.device('cpu')
model=SmallCNN(len(le.classes_)).to(device)
opt=optim.Adam(model.parameters(),lr=1e-3)
sched=optim.lr_scheduler.StepLR(opt,step_size=5,gamma=0.5)
crit=nn.CrossEntropyLoss()

EPOCHS=10
tr_accs,va_accs,tr_losses,va_losses=[],[],[],[]
best_acc,best_state=0,None
t0=time.time()
print(f"\nTraining {EPOCHS} epochs...")

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
    print(f"  Ep {ep+1:>2}/{EPOCHS}  tr={ta:.3f}  val={va:.3f}")

print(f"\nDone {time.time()-t0:.1f}s | best={best_acc:.3f}")

# Eval
model.load_state_dict(best_state); model.eval()
all_p,all_l=[],[]
with torch.no_grad():
    for imgs,lbls in te_dl:
        all_p.extend(model(imgs).argmax(1).numpy()); all_l.extend(lbls.numpy())

rpt=classification_report(all_l,all_p,target_names=le.classes_,output_dict=True)
cm=confusion_matrix(all_l,all_p)
print("\n"+classification_report(all_l,all_p,target_names=le.classes_))

# Save model
torch.save(best_state,os.path.join(OUT_DIR,"model_cnn_multi.pth"))

# Plots
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
ax1.plot(tr_losses,label='Train',c='#e74c3c'); ax1.plot(va_losses,label='Val',c='#3498db')
ax1.set_title("CNN Loss"); ax1.legend()
ax2.plot(tr_accs,label='Train',c='#e74c3c'); ax2.plot(va_accs,label='Val',c='#3498db')
ax2.set_title("CNN Accuracy"); ax2.legend()
plt.suptitle("CNN Training — PPE Detection",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"02_cnn_training_curves.png"),dpi=150,bbox_inches='tight')
plt.close()

fig,ax=plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay(cm,display_labels=le.classes_).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
ax.set_title(f"CNN Confusion Matrix\nacc={rpt['accuracy']:.3f}"); ax.tick_params(axis='x',rotation=35)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"08_cnn_confusion.png"),dpi=150,bbox_inches='tight')
plt.close()

# Append to summary
cnn_row={'Model':'CNN (multi-class)','Task':'multi',
         'Accuracy':round(rpt['accuracy'],4),
         'Macro F1':round(rpt['macro avg']['f1-score'],4),
         'Weighted F1':round(rpt['weighted avg']['f1-score'],4),
         'Train Time(s)':round(time.time()-t0,1)}
try:
    old=pd.read_csv(os.path.join(OUT_DIR,"model_summary_ml.csv"))
    full=pd.concat([old,pd.DataFrame([cnn_row])],ignore_index=True)
except:
    full=pd.DataFrame([cnn_row])
full.to_csv(os.path.join(OUT_DIR,"model_summary_all.csv"),index=False)
print("\nFull summary saved:")
print(full.to_string(index=False))
