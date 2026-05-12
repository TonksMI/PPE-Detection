"""One-shot: regenerate deeplab_pred_grid.png from saved checkpoint."""
import os, sys, numpy as np, torch, torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from ppe_deeplab_train import SegDataset

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES = 6
OUT_DIR   = str(Path(__file__).resolve().parent.parent / "results" / "models")

COLORS = np.array([
    [0,   0,   0  ], [128, 0,   128], [39,  174, 96 ],
    [231, 76,  60 ], [230, 126, 34 ], [41,  128, 185],
], dtype=np.uint8)
mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std_t  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[-1]     = nn.Conv2d(256, N_CLASSES, kernel_size=1)
model.aux_classifier[-1] = nn.Conv2d(256, N_CLASSES, kernel_size=1)
ckpt = torch.load(os.path.join(OUT_DIR, "deeplab_model.pth"), weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(DEVICE).eval()

val_ds    = SegDataset("val", augment=False)
n_samples = min(8, len(val_ds))
val_items = [val_ds[i] for i in range(n_samples)]

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
with torch.no_grad():
    for i in range(n_samples):
        img_t, _ = val_items[i]
        pred = model(img_t.unsqueeze(0).to(DEVICE))["out"].argmax(1)[0].cpu()
        img_np = ((img_t * std_t + mean_t).permute(1,2,0).numpy() * 255).clip(0,255).astype(np.uint8)
        row, base_col = divmod(i, 2)
        axes[row][base_col*2].imshow(img_np);     axes[row][base_col*2].axis("off");     axes[row][base_col*2].set_title("Image", fontsize=8)
        axes[row][base_col*2+1].imshow(COLORS[pred.numpy()]); axes[row][base_col*2+1].axis("off"); axes[row][base_col*2+1].set_title("Pred", fontsize=8)

plt.suptitle("DeepLabV3+ Predictions", fontsize=12)
plt.tight_layout()
out = os.path.join(OUT_DIR, "deeplab_pred_grid.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")
