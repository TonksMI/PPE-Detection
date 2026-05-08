"""
Replot Individual SVM and RF Heatmaps from Pickle
Always standardizes strictly isolated heatmaps via testing-only data splits.
"""
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def main():
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE_DIR = os.path.join(BASE, "cache")
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "models")
    
    # Load feature banks
    print("Loading test data caches...")
    X_ml = np.load(os.path.join(CACHE_DIR, "features_600.npy"))
    labels_raw = np.load(os.path.join(CACHE_DIR, "crops_y_600.npy"))
    
    # Recreate the exact le_multi mappings
    le_multi = joblib.load(os.path.join(OUT_DIR, "prod_le_multi.pkl"))
    y_multi = le_multi.transform(labels_raw)
    
    # 20% strictly testing split (matching global state seed)
    Xtr, Xte, ytr, yte = train_test_split(X_ml, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
    print(f"Test mapping restricted strictly to {len(Xte)} isolated instances.")
    
    models_to_run = [
        ('prod_svm_multi.pkl', 'SVM', 'prod_svm_confusion_single.png'),
        ('prod_rf_multi.pkl', 'Random Forest', 'prod_rf_confusion_single.png'),
    ]
    
    for mdl_file, name, out_file in models_to_run:
        print(f"Re-predicting {name} heatmaps against testing data...")
        model = joblib.load(os.path.join(OUT_DIR, mdl_file))
        yp = model.predict(Xte)
        
        cm = confusion_matrix(yte, yp)
        acc = (yp == yte).mean()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        dist = ConfusionMatrixDisplay(cm, display_labels=le_multi.classes_)
        dist.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
        ax.set_title(f"{name} Confusion Matrix (Testing Data)\nValidation Acc = {acc:.3f}")
        ax.tick_params(axis='x', rotation=35)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, out_file), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved cleanly detached {out_file}")

if __name__ == "__main__":
    main()
