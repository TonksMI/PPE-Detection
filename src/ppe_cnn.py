"""
PPE Detection — CNN Training (PyTorch, lightweight, fast)
15 epochs, 64x64 crops, 5 classes
"""
import os, glob, random, time, warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42); torch.manual_seed(42)

DATASET_DIR = "/sessions/sleepy-epic-pascal/datasets/helmet-safety-vest-detection-master"
IMG_DIR  = os.path.join(DATASET_DIR, "train-images-data")
ANN_DIR  = os.path.join(DATASET_DIR, "train-images-annotations-new")
OUT_DIR  = "/sessions/sleepy-epic-pascal/mnt/Computer Vision"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CLASS_MAP = {
    "helmet":"helmet", "safety vest":"safety_vest",
    "person with full safety":"full_ppe",
    "person with partial safety":"partial_ppe",
    "person without safety":"no_ppe",
}
BINARY_MAP = {
    "helmet":"ppe_present","safety_vest":"ppe_present",
    "full_ppe":"ppe_present","partial_ppe":"ppe_present","no_ppe":"no_ppe",
}

print("="*55); print("PPE CNN TRAINING"); print("="*55)

# ── Load metadata ──────────────────────────────────────────────
records = []
for xf in sorted(glob.glob(os.path.join(ANN_DIR, "*.xml"))):
    try:
        root = ET.parse(xf).getroot()
        fname = root.findtext("filename")
        img_path = os.path.join(IMG_DIR, fname) if fname else None
        if not img_path or not os.path.exists(img_path):
            stem = os.path.splitext(os.path.basename(xf))[0]
            for ext in [".jpg",".jpeg",".png"]:
                c = os.path.join(IMG_DIR, stem+ext)
                if os.path.exists(c): img_path=c; break
        if not img_path or not os.path.exists(img_path): continue
        sz = root.find("size"); iw,ih = int(sz.findtext("width",0)), int(sz.findtext("height",0))
        for obj in root.findall("object"):
            raw = obj.findtext("name","").strip().lower()
            if raw not in RAW_CLASS_MAP: continue
            cls = RAW_CLASS_MAP[raw]
            bb = obj.find("bndbox")
            x1=max(0,int(float(bb.findtext("xmin",0)))); y1=max(0,int(float(bb.findtext("ymin",0))))
            x2=min(iw,int(float(bb.findtext("xmax",iw)))); y2=min(ih,int(float(bb.findtext("ymax",ih))))
            if x2>x1 and y2>y1:
                records.append({"img_path":img_path,"class":cls,"x1":x1,"y1":y1,"x2":x2,"y2":y2})
    except: pass

df = pd.DataFrame(records)
# Balance classes
MAX_PER_CLASS = 400
df_bal = df.groupby('class').apply(lambda g: g.sample(min(len(g),MAX_PER_CLASS),random_state=42)).reset_index(drop=True)
print(f"Balanced dataset: {len(df_bal)} crops")
print(df_bal['class'].value_counts().to_string())

le = LabelEncoder()
df_bal['label'] = le.fit_transform(df_bal['class'])
tr_df, te_df = train_test_split(df_bal, test_size=0.2, random_state=42, stratify=df_bal['label'])
print(f"\nTrain={len(tr_df)} | Test={len(te_df)}")
print(f"Classes: {list(le.classes_)}")

# ── Dataset ────────────────────────────────────────────────────
class PPECropDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df=df.reset_index(drop=True); self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = cv2.imread(r['img_path'])
        if img is None: img=np.zeros((64,64,3),dtype=np.uint8)
        crop = img[r['y1']:r['y2'], r['x1']:r['x2']]
        if crop.size==0: crop=np.zeros((64,64,3),dtype=np.uint8)
        crop = cv2.cvtColor(cv2.resize(crop,(64,64)), cv2.COLOR_BGR2RGB)
        from PIL import Image
        crop_pil = Image.fromarray(crop)
        if self.transform: crop_pil=self.transform(crop_pil)
        return crop_pil, int(r['label'])

tf_tr = T.Compose([
    T.RandomHorizontalFlip(), T.RandomRotation(15),
    T.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.2),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
tf_te = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

tr_dl = DataLoader(PPECropDataset(tr_df,tf_tr), batch_size=32, shuffle=True,  num_workers=0)
te_dl = DataLoader(PPECropDataset(te_df,tf_te), batch_size=32, shuffle=False, num_workers=0)

# ── Model ──────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 32
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), # 16
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),# 8
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)),
        )
        self.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(256*4,256), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(256, nc))
    def forward(self,x): return self.fc(self.net(x).flatten(1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmallCNN(len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

EPOCHS = 15
tr_accs, va_accs, tr_losses, va_losses = [], [], [], []
best_acc, best_state = 0, None
t0 = time.time()

print(f"\nTraining {EPOCHS} epochs on {device}...")
for epoch in range(EPOCHS):
    model.train()
    tl, tc, tt = 0, 0, 0
    for imgs, lbls in tr_dl:
        imgs,lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs); loss = criterion(out,lbls)
        loss.backward(); optimizer.step()
        tl+=loss.item()*len(lbls); tc+=(out.argmax(1)==lbls).sum().item(); tt+=len(lbls)
    model.eval(); vl,vc,vt=0,0,0
    with torch.no_grad():
        for imgs,lbls in te_dl:
            imgs,lbls=imgs.to(device),lbls.to(device)
            out=model(imgs); loss=criterion(out,lbls)
            vl+=loss.item()*len(lbls); vc+=(out.argmax(1)==lbls).sum().item(); vt+=len(lbls)
    ta,va=tc/tt,vc/vt
    tr_accs.append(ta); va_accs.append(va)
    tr_losses.append(tl/tt); va_losses.append(vl/vt)
    scheduler.step()
    if va>best_acc: best_acc=va; best_state={k:v.clone() for k,v in model.state_dict().items()}
    if (epoch+1)%3==0: print(f"  Ep {epoch+1:>2}/{EPOCHS}  tr={ta:.3f}  val={va:.3f}")

print(f"Done in {time.time()-t0:.1f}s | best val={best_acc:.3f}")

# ── Eval best ─────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
all_p,all_l=[],[]
with torch.no_grad():
    for imgs,lbls in te_dl:
        all_p.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
        all_l.extend(lbls.numpy())

cnn_rpt = classification_report(all_l,all_p,target_names=le.classes_,output_dict=True)
cnn_cm  = confusion_matrix(all_l,all_p)
print(f"\nCNN Multi-class Results:")
print(classification_report(all_l,all_p,target_names=le.classes_))

# ── Save ───────────────────────────────────────────────────────
torch.save(best_state, os.path.join(OUT_DIR,"model_cnn_multi.pth"))

# ── Plots ──────────────────────────────────────────────────────
# Training curves
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
ax1.plot(tr_losses,label='Train',color='#e74c3c'); ax1.plot(va_losses,label='Val',color='#3498db')
ax1.set_title("CNN Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(tr_accs,label='Train',color='#e74c3c'); ax2.plot(va_accs,label='Val',color='#3498db')
ax2.set_title("CNN Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.suptitle("CNN Training History — PPE Detection",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"02_cnn_training_curves.png"),dpi=150,bbox_inches='tight')
plt.close(); print("Saved training curves")

# CNN Confusion matrix
fig,ax=plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay(cnn_cm,display_labels=le.classes_).plot(ax=ax,cmap='Blues',colorbar=False,values_format='d')
ax.set_title(f"CNN — acc={cnn_rpt['accuracy']:.3f}",fontsize=11); ax.tick_params(axis='x',rotation=35)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"08_cnn_confusion.png"),dpi=150,bbox_inches='tight')
plt.close(); print("Saved CNN confusion matrix")

# Save CNN results to append to model summary
cnn_row = {
    'Model': 'CNN (multi-class)', 'Task': 'multi',
    'Accuracy': round(cnn_rpt['accuracy'],4),
    'Macro F1': round(cnn_rpt['macro avg']['f1-score'],4),
    'Weighted F1': round(cnn_rpt['weighted avg']['f1-score'],4),
    'Train Time(s)': round(time.time()-t0,1)
}
# Append to existing summary
try:
    old = pd.read_csv(os.path.join(OUT_DIR,"model_summary_ml.csv"))
    full = pd.concat([old, pd.DataFrame([cnn_row])], ignore_index=True)
except:
    full = pd.DataFrame([cnn_row])
full.to_csv(os.path.join(OUT_DIR,"model_summary_all.csv"),index=False)
print("\nFull model summary:")
print(full.to_string(index=False))
