"""
ViT Architecture Training for PPE Classification
Uses torchvision's ViT-B-16 pretrained model.
"""
import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ViT_B_16_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description='PPE ViT Training')
    parser.add_argument('--epochs', type=int, default=20, help='ViT training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH = args.batch_size
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Paths
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.exists("D:/datasets/ppe_crops"):
        DATA_DIR = "D:/datasets/ppe_crops"
    else:
        DATA_DIR = os.path.join(BASE, "datasets", "ppe_crops")
        
    OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*65)
    print("PPE ViT CLASSIFICATION TRAINING")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH}")
    print("="*65)
    
    # 1. Dataset & DataLoaders
    # ViT expects 224x224 images and specific normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading data from {DATA_DIR}...")
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist. Please run setup_datasets.py first.")
        return

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)

    classes = train_data.classes
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    train_dl = DataLoader(train_data, batch_size=BATCH, shuffle=True, pin_memory=(DEVICE.type=='cuda'))
    val_dl = DataLoader(val_data, batch_size=BATCH, shuffle=False, pin_memory=(DEVICE.type=='cuda'))

    # 2. Model Setup
    print("Initializing ViT-B/16 model...")
    # Load pretrained ViT
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    
    # Replace classification head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    # 3. Optimizer & Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # Fine-tuning: lower learning rate for base, slightly higher for head
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not 'heads' in n], 'lr': 1e-5},
        {'params': model.heads.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Training Loop
    best_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    t_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            tl += loss.item() * len(lbls)
            tc += (out.argmax(1) == lbls).sum().item()
            tt += len(lbls)
            
        scheduler.step()
        
        # Validation
        model.eval()
        vl, vc, vt = 0.0, 0, 0
        
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs)
                vl += criterion(out, lbls).item() * len(lbls)
                vc += (out.argmax(1) == lbls).sum().item()
                vt += len(lbls)
                
        ta, va = tc / tt, vc / vt
        history['train_loss'].append(tl / tt)
        history['val_loss'].append(vl / vt)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)
        
        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        print(f"Ep {epoch:>2}/{EPOCHS} | Train Acc: {ta:.3f} Loss: {tl/tt:.3f} | Val Acc: {va:.3f} Loss: {vl/vt:.3f}")

    total_time = time.time() - t_start
    print(f"\nTraining Complete in {total_time/60:.1f}m | Best Val Acc: {best_acc:.3f}")

    # 5. Evaluate and Save
    model.load_state_dict(best_state)
    model.eval()
    
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, lbls in val_dl:
            out = model(imgs.to(DEVICE))
            all_p.extend(out.argmax(1).cpu().numpy())
            all_l.extend(lbls.numpy())
            
    print("\nViT Classification Report:")
    print(classification_report(all_l, all_p, target_names=classes))
    
    # Save model weights
    torch.save({
        'state_dict': best_state,
        'classes': classes,
        'epoch': EPOCHS,
        'best_val_acc': best_acc,
        'arch': 'ViT-B-16'
    }, os.path.join(OUT_DIR, "prod_vit_model.pth"))
    
    # Plot history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ep_r = range(1, EPOCHS + 1)
    
    ax1.plot(ep_r, history['train_loss'], 'r-', label='Train')
    ax1.plot(ep_r, history['val_loss'], 'b-', label='Val')
    ax1.set_title("ViT Loss")
    ax1.grid(alpha=0.3); ax1.legend()
    
    ax2.plot(ep_r, history['train_acc'], 'r-', label='Train')
    ax2.plot(ep_r, history['val_acc'], 'b-', label='Val')
    ax2.axhline(best_acc, color='g', linestyle='--', label=f'Best: {best_acc:.3f}')
    ax2.set_title("ViT Accuracy")
    ax2.grid(alpha=0.3); ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prod_vit_training.png"), dpi=150)
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(all_l, all_p)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('ViT Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prod_vit_confusion.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
