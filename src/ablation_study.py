"""
Ablation Study
Runs a blank model from scratch (Condition B)
And a frozen transfer model (Condition C)
Using ResNet18 for fast tracking metrics
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.exists("D:/datasets/ppe_crops"):
    DATA_DIR = "D:/datasets/ppe_crops"
else:
    DATA_DIR = os.path.join(BASE, "datasets", "ppe_crops")

def run_experiment(condition_name, weights=None, freeze=False, epochs=5):
    print(f"\n--- Running Experiment: {condition_name} ---")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)
    
    # Use smaller batch for safety
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Load model
    model = models.resnet18(weights=weights)
    
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
            
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5) # 5 classes for PPE
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    # If frozen, only the FC layer has requires_grad=True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for imgs, lbls in train_dl:
            optimizer.zero_grad()
            out = model(imgs.to(DEVICE))
            loss = criterion(out, lbls.to(DEVICE))
            loss.backward()
            optimizer.step()
            
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                out = model(imgs.to(DEVICE))
                vc += (out.argmax(1) == lbls.to(DEVICE)).sum().item()
                vt += len(lbls)
                
        val_acc = vc / vt
        best_acc = max(best_acc, val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.4f}")
        
    return best_acc

def main():
    # Condition B: Blank model from scratch
    acc_blank = run_experiment("Blank Model from Scratch (ResNet18)", weights=None, freeze=False, epochs=5)
    
    # Condition C: Frozen transfer model
    acc_frozen = run_experiment("Frozen Transfer Model (ResNet18)", weights=models.ResNet18_Weights.DEFAULT, freeze=True, epochs=5)
    
    print("\n================== RESULTS ==================")
    print(f"(B) Blank Model From Scratch  : {acc_blank*100:.2f}%")
    print(f"(C) Frozen Transfer Model     : {acc_frozen*100:.2f}%")
    
    # Save results to a text file
    with open('ablation_results.txt', 'w') as f:
        f.write(f"Blank from scratch: {acc_blank*100:.2f}%\n")
        f.write(f"Frozen transfer: {acc_frozen*100:.2f}%\n")

if __name__ == "__main__":
    main()
