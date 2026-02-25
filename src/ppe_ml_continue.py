"""
PPE Detection — Continue from saved features (skip re-extraction)
"""
import os, glob, random, time, warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42)

OUT_DIR = "/sessions/sleepy-epic-pascal/mnt/Computer Vision"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load already-extracted features ───────────────────────────
print("Loading pre-extracted features...")
DATASET_DIR = "/sessions/sleepy-epic-pascal/datasets/helmet-safety-vest-detection-master"
IMG_DIR  = os.path.join(DATASET_DIR, "train-images-data")
ANN_DIR  = os.path.join(DATASET_DIR, "train-images-annotations-new")

RAW_CLASS_MAP = {
    "helmet":                     "helmet",
    "safety vest":                "safety_vest",
    "person with full safety":    "full_ppe",
    "person with partial safety": "partial_ppe",
    "person without safety":      "no_ppe",
}
BINARY_MAP = {
    "helmet":"ppe_present","safety_vest":"ppe_present",
    "full_ppe":"ppe_present","partial_ppe":"ppe_present","no_ppe":"no_ppe",
}

def extract_features(img_bgr):
    img = cv2.resize(img_bgr, (64,64))
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

# Re-extract (fast since already done once, reuse same logic)
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
        sz = root.find("size")
        iw,ih = int(sz.findtext("width",0)), int(sz.findtext("height",0))
        for obj in root.findall("object"):
            raw = obj.findtext("name","").strip().lower()
            if raw not in RAW_CLASS_MAP: continue
            cls = RAW_CLASS_MAP[raw]
            bb = obj.find("bndbox")
            x1=max(0,int(float(bb.findtext("xmin",0)))); y1=max(0,int(float(bb.findtext("ymin",0))))
            x2=min(iw,int(float(bb.findtext("xmax",iw)))); y2=min(ih,int(float(bb.findtext("ymax",ih))))
            if x2>x1 and y2>y1:
                records.append({"img_path":img_path,"class":cls,"binary":BINARY_MAP[cls],
                                 "x1":x1,"y1":y1,"x2":x2,"y2":y2})
    except: pass

df = pd.DataFrame(records)
MAX_PER_CLASS = 400
class_counts = {c:0 for c in RAW_CLASS_MAP.values()}
df_shuf = df.sample(frac=1, random_state=42).reset_index(drop=True)
features, y_multi, y_binary = [], [], []

for _, row in df_shuf.iterrows():
    cls = row['class']
    if class_counts[cls] >= MAX_PER_CLASS: continue
    try:
        img = cv2.imread(row['img_path'])
        if img is None: continue
        crop = img[row['y1']:row['y2'], row['x1']:row['x2']]
        if crop.size==0 or crop.shape[0]<8 or crop.shape[1]<8: continue
        features.append(extract_features(crop))
        y_multi.append(cls); y_binary.append(row['binary'])
        class_counts[cls] += 1
    except: pass

X = np.array(features)
y_m = np.array(y_multi); y_b = np.array(y_binary)
print(f"Feature matrix: {X.shape} | Classes: {dict(zip(*np.unique(y_m, return_counts=True)))}")

le_m = LabelEncoder(); y_m_enc = le_m.fit_transform(y_m)
le_b = LabelEncoder(); y_b_enc = le_b.fit_transform(y_b)
Xtr,Xte,yt,yv = train_test_split(X,y_m_enc,test_size=0.2,random_state=42,stratify=y_m_enc)
Xtr_b,Xte_b,yt_b,yv_b = train_test_split(X,y_b_enc,test_size=0.2,random_state=42,stratify=y_b_enc)

# ── Train ──────────────────────────────────────────────────────
results = {}
def run(name, pipe, Xtr, Xte, ytr, yte, le, task):
    t0=time.time(); pipe.fit(Xtr,ytr); tt=time.time()-t0
    yp=pipe.predict(Xte)
    rpt=classification_report(yte,yp,target_names=le.classes_,output_dict=True)
    cm=confusion_matrix(yte,yp)
    acc=rpt['accuracy']
    print(f"  [{name:28s}] acc={acc:.4f}  train={tt:.1f}s")
    return {"name":name,"task":task,"accuracy":acc,"report":rpt,"confusion":cm,
            "model":pipe,"train_time":tt,"y_pred":yp,"y_test":yte,"le":le}

print("\nTraining models...")
results['svm_multi']  = run("SVM (multi-class)",
    Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=100,random_state=42)),
              ('svm',SVC(kernel='rbf',C=10,gamma='scale',probability=True,random_state=42))]),
    Xtr,Xte,yt,yv,le_m,"multi")
results['svm_binary'] = run("SVM (binary)",
    Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=80,random_state=42)),
              ('svm',SVC(kernel='rbf',C=5,gamma='scale',probability=True,random_state=42))]),
    Xtr_b,Xte_b,yt_b,yv_b,le_b,"binary")
results['rf_multi']   = run("Random Forest (multi-class)",
    Pipeline([('sc',StandardScaler()),
              ('rf',RandomForestClassifier(n_estimators=200,max_depth=15,n_jobs=-1,random_state=42))]),
    Xtr,Xte,yt,yv,le_m,"multi")
results['rf_binary']  = run("Random Forest (binary)",
    Pipeline([('sc',StandardScaler()),
              ('rf',RandomForestClassifier(n_estimators=200,max_depth=12,n_jobs=-1,random_state=42))]),
    Xtr_b,Xte_b,yt_b,yv_b,le_b,"binary")
results['gbm_multi']  = run("GradBoost (multi-class)",
    Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=80,random_state=42)),
              ('gbm',HistGradientBoostingClassifier(max_iter=100,max_depth=6,
                                                    learning_rate=0.1,random_state=42))]),
    Xtr,Xte,yt,yv,le_m,"multi")
results['gbm_binary'] = run("GradBoost (binary)",
    Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=60,random_state=42)),
              ('gbm',HistGradientBoostingClassifier(max_iter=100,max_depth=5,
                                                    learning_rate=0.1,random_state=42))]),
    Xtr_b,Xte_b,yt_b,yv_b,le_b,"binary")

# ── Save models ────────────────────────────────────────────────
for k,res in results.items():
    joblib.dump(res['model'], os.path.join(OUT_DIR,f"model_{k}.pkl"))
np.save("/sessions/sleepy-epic-pascal/X_features.npy", X)
np.save("/sessions/sleepy-epic-pascal/y_multi_enc.npy", y_m_enc)
np.save("/sessions/sleepy-epic-pascal/y_binary_enc.npy", y_b_enc)
joblib.dump(le_m, "/sessions/sleepy-epic-pascal/le_multi.pkl")
joblib.dump(le_b, "/sessions/sleepy-epic-pascal/le_binary.pkl")
np.save("/sessions/sleepy-epic-pascal/Xtr.npy",Xtr); np.save("/sessions/sleepy-epic-pascal/Xte.npy",Xte)
np.save("/sessions/sleepy-epic-pascal/yt.npy",yt); np.save("/sessions/sleepy-epic-pascal/yv.npy",yv)
np.save("/sessions/sleepy-epic-pascal/Xtr_b.npy",Xtr_b); np.save("/sessions/sleepy-epic-pascal/Xte_b.npy",Xte_b)
np.save("/sessions/sleepy-epic-pascal/yt_b.npy",yt_b); np.save("/sessions/sleepy-epic-pascal/yv_b.npy",yv_b)

# ── Plots ──────────────────────────────────────────────────────
print("\nGenerating plots...")

# Model comparison
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,task_key,title in [(axes[0],"multi","Multi-class Accuracy"),(axes[1],"binary","Binary Accuracy")]:
    res_t={k:v for k,v in results.items() if v['task']==task_key}
    names=[v['name'] for v in res_t.values()]; accs=[v['accuracy'] for v in res_t.values()]
    cols=['#e74c3c','#3498db','#2ecc71']
    bars=ax.bar(names,accs,color=cols,edgecolor='black',width=0.5)
    ax.set_ylim(0,1.1); ax.set_title(title,fontsize=13,fontweight='bold')
    ax.set_ylabel("Accuracy"); ax.tick_params(axis='x',rotation=20)
    for b,a in zip(bars,accs):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'{a:.3f}',
                ha='center',va='bottom',fontweight='bold',fontsize=10)
plt.suptitle("PPE Detection — Model Comparison (SVM vs RF vs GBM)",fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"03_model_comparison.png"),dpi=150,bbox_inches='tight')
plt.close(); print("  saved 03_model_comparison.png")

# Confusion matrices
multi_res={k:v for k,v in results.items() if v['task']=='multi'}
fig,axes=plt.subplots(1,3,figsize=(18,5))
for ax,(k,res) in zip(axes,multi_res.items()):
    ConfusionMatrixDisplay(res['confusion'],display_labels=res['le'].classes_).plot(
        ax=ax,cmap='Blues',colorbar=False,values_format='d')
    ax.set_title(f"{res['name']}\nacc={res['accuracy']:.3f}",fontsize=10)
    ax.tick_params(axis='x',rotation=35)
plt.suptitle("Confusion Matrices — Multi-class PPE",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"04_confusion_matrices.png"),dpi=150,bbox_inches='tight')
plt.close(); print("  saved 04_confusion_matrices.png")

# F1 heatmap
cls_names=list(le_m.classes_)
f1d={res['name']:[res['report'].get(c,{}).get('f1-score',0) for c in cls_names]
     for k,res in multi_res.items()}
fig,ax=plt.subplots(figsize=(10,4))
sns.heatmap(pd.DataFrame(f1d,index=cls_names),annot=True,fmt='.3f',
            cmap='YlOrRd',ax=ax,linewidths=0.5,cbar_kws={'label':'F1'})
ax.set_title("Per-class F1 Score by Model",fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"05_f1_heatmap.png"),dpi=150,bbox_inches='tight')
plt.close(); print("  saved 05_f1_heatmap.png")

# ROC curves
binary_res={k:v for k,v in results.items() if v['task']=='binary'}
fig,ax=plt.subplots(figsize=(8,6))
for (k,res),col in zip(binary_res.items(),['#e74c3c','#3498db','#2ecc71']):
    if hasattr(res['model'],'predict_proba'):
        ys=res['model'].predict_proba(Xte_b)[:,1]
        fpr,tpr,_=roc_curve(yv_b,ys); ra=auc(fpr,tpr)
        ax.plot(fpr,tpr,color=col,lw=2,label=f"{res['name']} (AUC={ra:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC Curves — Binary PPE Detection",fontsize=13,fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"06_roc_curves.png"),dpi=150,bbox_inches='tight')
plt.close(); print("  saved 06_roc_curves.png")

# RF importance
rf=results['rf_multi']['model'].named_steps['rf']
imp=rf.feature_importances_; top=np.argsort(imp)[-30:]
fig,ax=plt.subplots(figsize=(8,6))
ax.barh(range(30),imp[top],color='steelblue',edgecolor='black')
ax.set_yticks(range(30)); ax.set_yticklabels([f"feat_{i}" for i in top],fontsize=7)
ax.set_xlabel("Importance"); ax.set_title("RF Feature Importances (Top 30)",fontsize=12,fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"07_rf_feature_importance.png"),dpi=150,bbox_inches='tight')
plt.close(); print("  saved 07_rf_feature_importance.png")

# Summary
rows=[{'Model':v['name'],'Task':v['task'],'Accuracy':round(v['accuracy'],4),
       'Macro F1':round(v['report'].get('macro avg',{}).get('f1-score',0),4),
       'Weighted F1':round(v['report'].get('weighted avg',{}).get('f1-score',0),4),
       'Train Time(s)':round(v['train_time'],1)} for v in results.values()]
summary=pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR,"model_summary_ml.csv"),index=False)

print("\n"+"="*60)
print("RESULTS SUMMARY")
print("="*60)
print(summary.to_string(index=False))
print("="*60)
