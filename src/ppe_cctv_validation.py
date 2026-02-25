"""
PPE CCTV Validation Script
Runs trained models on surveillance-style images
Uses sliding window detection for full-scene images
"""
import os, glob, cv2, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import joblib
import torch, torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

OUT_DIR  = "/sessions/sleepy-epic-pascal/mnt/Computer Vision"
VAL_DIR  = "/sessions/sleepy-epic-pascal/cctv_validation"

# ── Load SVM model (best single model) ────────────────────────
print("Loading trained models...")
svm_model = joblib.load(os.path.join(OUT_DIR, "model_svm_multi.pkl"))
rf_model  = joblib.load(os.path.join(OUT_DIR, "model_rf_multi.pkl"))

# ── Reload label encoder ──────────────────────────────────────
import glob as glob_mod, xml.etree.ElementTree as ET
DATASET_DIR = "/sessions/sleepy-epic-pascal/datasets/helmet-safety-vest-detection-master"
ANN_DIR = os.path.join(DATASET_DIR, "train-images-annotations-new")
IMG_DIR = os.path.join(DATASET_DIR, "train-images-data")
RAW_CLASS_MAP = {"helmet":"helmet","safety vest":"safety_vest",
    "person with full safety":"full_ppe","person with partial safety":"partial_ppe",
    "person without safety":"no_ppe"}
labels_for_le = []
for xf in sorted(glob_mod.glob(os.path.join(ANN_DIR,"*.xml")))[:200]:
    try:
        for obj in ET.parse(xf).getroot().findall("object"):
            raw = obj.findtext("name","").strip().lower()
            if raw in RAW_CLASS_MAP: labels_for_le.append(RAW_CLASS_MAP[raw])
    except: pass
le = LabelEncoder(); le.fit(labels_for_le)
CLASS_NAMES = list(le.classes_)
print(f"Classes: {CLASS_NAMES}")

# ── Reload CNN ─────────────────────────────────────────────────
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

cnn = SmallCNN(len(CLASS_NAMES))
state = torch.load(os.path.join(OUT_DIR,"model_cnn_multi.pth"), map_location='cpu')
cnn.load_state_dict(state); cnn.eval()

# ── Feature extraction (for SVM/RF) ──────────────────────────
def extract_features(img_bgr):
    img = cv2.resize(img_bgr,(64,64))
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
    return np.concatenate([hog_f]+color_f).reshape(1,-1)

def cnn_predict(img_bgr):
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr,(64,64)), cv2.COLOR_BGR2RGB)
    x = torch.tensor(img_rgb.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    x=(x-mean)/std
    with torch.no_grad():
        out=cnn(x); probs=torch.softmax(out,1).squeeze().numpy()
    return CLASS_NAMES[probs.argmax()], probs

# Color map for classes
CLASS_COLORS = {
    'helmet':      '#27ae60',
    'safety_vest': '#2980b9',
    'full_ppe':    '#8e44ad',
    'partial_ppe': '#e67e22',
    'no_ppe':      '#e74c3c',
}

# ── Sliding window PPE scanning ───────────────────────────────
def sliding_window_detect(img_bgr, win_sizes=[(100,100),(150,150),(200,200)],
                           stride=50, conf_threshold=0.55):
    """Apply sliding window across image, collect high-confidence detections"""
    h,w = img_bgr.shape[:2]
    detections = []
    for wh,ww in win_sizes:
        for y in range(0, h-wh+1, stride):
            for x in range(0, w-ww+1, stride):
                crop = img_bgr[y:y+wh, x:x+ww]
                cls, probs = cnn_predict(crop)
                conf = probs.max()
                if conf >= conf_threshold:
                    detections.append({'x':x,'y':y,'w':ww,'h':wh,
                                        'class':cls,'conf':float(conf),
                                        'probs':probs})
    return detections

def nms(detections, iou_threshold=0.4):
    """Simple NMS to reduce overlapping boxes"""
    if not detections: return []
    detections = sorted(detections, key=lambda d: d['conf'], reverse=True)
    kept = []
    for det in detections:
        x1,y1 = det['x'],det['y']; x2,y2=x1+det['w'],y1+det['h']
        overlap = False
        for k in kept:
            kx1,ky1=k['x'],k['y']; kx2,ky2=kx1+k['w'],ky1+k['h']
            ix1=max(x1,kx1);iy1=max(y1,ky1);ix2=min(x2,kx2);iy2=min(y2,ky2)
            if ix2>ix1 and iy2>iy1:
                inter=(ix2-ix1)*(iy2-iy1)
                union=(x2-x1)*(y2-y1)+(kx2-kx1)*(ky2-ky1)-inter
                if inter/union > iou_threshold: overlap=True; break
        if not overlap: kept.append(det)
    return kept

# ── Process validation images ──────────────────────────────────
print("\nProcessing CCTV validation images...")
val_images = sorted(glob.glob(os.path.join(VAL_DIR,"*.jpg")) +
                    glob.glob(os.path.join(VAL_DIR,"*.JPG")) +
                    glob.glob(os.path.join(VAL_DIR,"*.png")))
print(f"Found {len(val_images)} validation images")

# Process up to 12 images for visualization
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

summary_data = []
for i, img_path in enumerate(val_images[:12]):
    img = cv2.imread(img_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = axes[i]
    ax.imshow(img_rgb)

    # For small images (crops), classify directly
    h,w = img.shape[:2]
    if h < 200 and w < 200:
        cls, probs = cnn_predict(img)
        conf = probs.max()
        color = CLASS_COLORS.get(cls, 'gray')
        rect = patches.Rectangle((2,2), w-4, h-4, lw=3,
                                   edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"{os.path.basename(img_path)}\n{cls} ({conf:.2f})",
                     fontsize=8, color=color)
        summary_data.append({'image':os.path.basename(img_path),
                              'type':'direct_crop','prediction':cls,
                              'confidence':round(float(conf),3),'detections':1})
    else:
        # Full scene: sliding window
        t0=time.time()
        dets = sliding_window_detect(img, win_sizes=[(80,80),(120,80),(100,120)],
                                      stride=40, conf_threshold=0.60)
        dets = nms(dets, iou_threshold=0.35)
        elapsed = time.time()-t0
        for det in dets[:8]:  # show up to 8 per image
            color = CLASS_COLORS.get(det['class'],'gray')
            rect = patches.Rectangle((det['x'],det['y']), det['w'], det['h'],
                                       lw=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            ax.text(det['x'], det['y']-3, f"{det['class'][:8]} {det['conf']:.2f}",
                    color='white', fontsize=6, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.8))
        label = f"Scene ({len(dets)} detections)"
        ax.set_title(f"{os.path.basename(img_path)}\n{label}", fontsize=8)
        # class summary
        cls_counts = {}
        for d in dets:
            cls_counts[d['class']] = cls_counts.get(d['class'],0)+1
        summary_data.append({'image':os.path.basename(img_path),
                              'type':'full_scene','prediction':str(cls_counts),
                              'confidence': round(np.mean([d['conf'] for d in dets]) if dets else 0, 3),
                              'detections':len(dets)})
    ax.axis('off')

# Hide unused axes
for j in range(len(val_images), 12):
    axes[j].axis('off')

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0],color=v,lw=3,label=k) for k,v in CLASS_COLORS.items()]
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5,-0.02))
plt.suptitle("PPE Detection — CCTV Validation Results\n(Surveillance-style images from multiple sources)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"09_cctv_validation.png"), dpi=150, bbox_inches='tight')
plt.close(); print("Saved 09_cctv_validation.png")

# ── Per-class accuracy on held-out images ─────────────────────
print("\nRunning per-class evaluation on image crops (direct classification)...")
crop_predictions = []
for img_path in val_images:
    img = cv2.imread(img_path)
    if img is None: continue
    h,w = img.shape[:2]
    if h<200 and w<200:
        cls_svm = le.inverse_transform(svm_model.predict(extract_features(img)))[0]
        cls_cnn, probs_cnn = cnn_predict(img)
        crop_predictions.append({
            'image': os.path.basename(img_path),
            'SVM_prediction': cls_svm,
            'CNN_prediction': cls_cnn,
            'CNN_confidence': round(float(probs_cnn.max()),3),
        })

if crop_predictions:
    pred_df = __import__('pandas').DataFrame(crop_predictions)
    pred_df.to_csv(os.path.join(OUT_DIR,"cctv_predictions.csv"), index=False)
    print(pred_df.to_string(index=False))

# ── Summary ────────────────────────────────────────────────────
import pandas as pd
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(OUT_DIR,"cctv_validation_summary.csv"), index=False)
print(f"\nProcessed {len(summary_data)} images")
print(f"Average detections per scene: {summary_df['detections'].mean():.1f}")
print(f"Average confidence: {summary_df['confidence'].mean():.3f}")
print("\nCCTV validation complete!")
