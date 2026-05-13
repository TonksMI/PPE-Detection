"""ppe_seg_qualitative.py

Generates best/worst-3 prediction grids with Grad-CAM and saliency maps for:
  - DeepLabV3+ ResNet50  (keremberke semantic seg, 11 classes)
  - YOLOv8n-seg          (keremberke instance seg, 10 classes)

Each figure layout (3 rows × 5 cols):
  Input | GT | Prediction | Grad-CAM overlay | Saliency overlay

Outputs → results/models/
  deeplab_best3.png   deeplab_worst3.png
  yolo_best3.png      yolo_worst3.png
"""

import os, sys, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
for _cand in [
    "D:/datasets/ppe_seg_ke",
    "D:/Claude/datasets/ppe_seg_ke",
    str(BASE.parent / "datasets" / "ppe_seg_ke"),
]:
    if os.path.exists(os.path.join(_cand, "semantic")):
        SEG_ROOT = Path(_cand)
        break
else:
    raise FileNotFoundError("ppe_seg_ke not found — run ppe_seg_keremberke_rebuild.py first")

OUT_DIR = BASE / "results" / "models"
MDL_DIR = OUT_DIR

# Model weights — check main results dir first, then any worktree
_deeplab_cands = [MDL_DIR / "deeplab_model.pth"] + sorted(
    BASE.glob(".claude/worktrees/*/results/models/deeplab_model.pth"),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
DEEPLAB_PT = next((p for p in _deeplab_cands if p.exists()), MDL_DIR / "deeplab_model.pth")

_yolo_cands = [MDL_DIR / "yolo_seg_best.pt"] + sorted(
    BASE.glob(".claude/worktrees/*/results/models/yolo_seg_best.pt"),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
YOLO_PT = next((p for p in _yolo_cands if p.exists()), MDL_DIR / "yolo_seg_best.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"DeepLab weights: {DEEPLAB_PT}")
print(f"YOLO weights:    {YOLO_PT}")

# ── Class definitions ──────────────────────────────────────────────────────────
CLASSES = [
    "background", "helmet", "no_helmet", "glove", "no_glove",
    "goggles", "no_goggles", "mask", "no_mask", "shoes", "no_shoes",
]
N_CLS = len(CLASSES)   # 11

YOLO_CLASSES = CLASSES[1:]   # 10 (no background)

# Distinct colours per class (RGB)
COLORS = [
    (  0,   0,   0),   # background  — black
    (255, 165,   0),   # helmet      — orange
    (128,   0, 128),   # no_helmet   — purple
    (  0, 200,   0),   # glove       — green
    (255,   0, 255),   # no_glove    — magenta
    (  0, 220, 220),   # goggles     — cyan
    (255, 140,   0),   # no_goggles  — dark-orange
    (220, 220,   0),   # mask        — yellow
    (128, 128,   0),   # no_mask     — olive
    ( 30,  30, 255),   # shoes       — blue
    (255,  30,  30),   # no_shoes    — red
]

# ── Image pre/post processing ──────────────────────────────────────────────────
IMG_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_tfm = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def preprocess(img_pil):
    """PIL image → normalised (1,3,H,W) CUDA tensor."""
    return _tfm(img_pil).unsqueeze(0).to(DEVICE)


def denorm(t):
    """(3,H,W) normalised tensor → (H,W,3) uint8 numpy."""
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    x = t.cpu() * s + m
    return (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def mask_to_rgb(mask_np):
    """Class-index H×W array → (H,W,3) uint8 RGB."""
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for c, col in enumerate(COLORS):
        rgb[mask_np == c] = col
    return rgb


def resize_map(arr, h, w):
    """Resize a float32 2-D map to (h, w)."""
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def apply_heatmap(cam, img_rgb):
    """Overlay a [0,1] float32 cam on an HWC uint8 image."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (0.5 * img_rgb.astype(np.float32) +
            0.5 * heat.astype(np.float32)).astype(np.uint8)


# ── Generic Grad-CAM ───────────────────────────────────────────────────────────
class GradCAM:
    """Register hooks on `target_layer`, compute weighted activation map."""

    def __init__(self, target_layer):
        self.acts  = None
        self.grads = None
        self._fh = target_layer.register_forward_hook(self._fhook)
        self._bh = target_layer.register_full_backward_hook(self._bhook)

    def _fhook(self, _m, _i, out):
        self.acts = out.detach()

    def _bhook(self, _m, _gi, grad_out):
        self.grads = grad_out[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def cam(self, score_tensor):
        """Call after forward; pass the scalar score to backward."""
        score_tensor.backward(retain_graph=True)
        w   = self.grads.mean(dim=(2, 3), keepdim=True)   # global avg-pool
        cam = F.relu((w * self.acts).sum(dim=1)).squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def vanilla_saliency(img_tensor, score_fn):
    """
    Vanilla gradient saliency: |∂score / ∂input|  max-pooled over channels.
    img_tensor should NOT already require grad — a fresh clone is used.
    """
    x = img_tensor.clone().detach().requires_grad_(True)
    score = score_fn(x)
    score.backward()
    sal = x.grad.abs().max(dim=1)[0].squeeze().cpu().numpy()
    sal /= sal.max() + 1e-8
    return sal


# ── DeepLabV3+ ────────────────────────────────────────────────────────────────
def load_deeplab():
    import torch.nn as nn
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    # Mirror training setup: start from pretrained weights, replace both classifier heads
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[-1]     = nn.Conv2d(256, N_CLS, kernel_size=1)
    model.aux_classifier[-1] = nn.Conv2d(256, N_CLS, kernel_size=1)
    ckpt  = torch.load(str(DEEPLAB_PT), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return model.to(DEVICE).eval()


def deeplab_pred(model, t):
    """Returns predicted class-index H×W numpy array (at input resolution)."""
    with torch.no_grad():
        out = model(t)["out"]          # (1,11,512,512)
    return out.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)


def image_miou(pred, gt, ignore_bg=True):
    """Mean IoU over classes present in GT (optionally ignoring background)."""
    start = 1 if ignore_bg else 0
    ious  = []
    for c in range(start, N_CLS):
        inter = int(((pred == c) & (gt == c)).sum())
        union = int(((pred == c) | (gt == c)).sum())
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def deeplab_gradcam_saliency(model, img_pil):
    """
    Returns (cam_overlay, sal_overlay) as HWC uint8 at original image size.
    Score: sum of predicted non-background class probabilities.
    Target layer: model.backbone.layer4  (ResNet last block, (1,2048,32,32))
    """
    t      = preprocess(img_pil)
    img_np = denorm(t.squeeze(0))
    H, W   = img_np.shape[:2]

    def score_fn(x):
        out = model(x)["out"]                   # (1,11,H,W)
        return out.softmax(dim=1)[:, 1:].sum()  # sum non-bg probs

    # --- Grad-CAM ---
    gc = GradCAM(model.backbone.layer4)
    model.zero_grad()
    with torch.enable_grad():
        t_g = preprocess(img_pil).requires_grad_(True)
        score = score_fn(t_g)
        cam_raw = gc.cam(score)
    gc.remove()
    cam_overlay = apply_heatmap(resize_map(cam_raw, H, W), img_np)

    # --- Saliency ---
    model.zero_grad()
    with torch.enable_grad():
        sal_raw = vanilla_saliency(preprocess(img_pil), score_fn)
    sal_overlay = apply_heatmap(resize_map(sal_raw, H, W), img_np)

    return cam_overlay, sal_overlay


def rank_deeplab(model, val_img_dir, val_msk_dir, n=3):
    fnames = sorted(os.listdir(val_img_dir))
    scores = []
    print(f"  Ranking {len(fnames)} val images for DeepLabV3+...")
    for i, fn in enumerate(fnames):
        if i % 100 == 0:
            print(f"    {i}/{len(fnames)}", end="\r", flush=True)
        img = Image.open(val_img_dir / fn).convert("RGB")
        gt  = np.array(Image.open(val_msk_dir / (Path(fn).stem + ".png")))
        t   = preprocess(img)
        pred = deeplab_pred(model, t)
        pred_up = cv2.resize(pred.astype(np.float32), (gt.shape[1], gt.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(np.int32)
        scores.append((image_miou(pred_up, gt.astype(np.int32)), fn))
    print()
    scores.sort(key=lambda x: x[0])
    worst = [s[1] for s in scores[:n]]
    best  = [s[1] for s in scores[-n:][::-1]]
    print(f"  Best  IoU: {[round(s[0],3) for s in scores[-n:][::-1]]}")
    print(f"  Worst IoU: {[round(s[0],3) for s in scores[:n]]}")
    return best, worst


def plot_deeplab(model, fnames, title, out_path):
    val_img_dir = SEG_ROOT / "semantic" / "val" / "images"
    val_msk_dir = SEG_ROOT / "semantic" / "val" / "masks"

    n    = len(fnames)
    cols = ["Input", "GT Mask", "Prediction", "Grad-CAM", "Saliency"]
    fig, axes = plt.subplots(n, 5, figsize=(22, 4.5 * n))
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for j, ct in enumerate(cols):
        axes[0, j].set_title(ct, fontsize=10, pad=4)

    for i, fn in enumerate(fnames):
        img_pil = Image.open(val_img_dir / fn).convert("RGB")
        gt      = np.array(Image.open(val_msk_dir / (Path(fn).stem + ".png")))
        t       = preprocess(img_pil)
        pred    = deeplab_pred(model, t)

        img_np  = np.array(img_pil.resize((gt.shape[1], gt.shape[0])))
        pred_up = cv2.resize(pred.astype(np.float32),
                             (gt.shape[1], gt.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(np.int32)

        iou = image_miou(pred_up, gt.astype(np.int32))
        cam_ov, sal_ov = deeplab_gradcam_saliency(model, img_pil)

        for j, im in enumerate([img_np,
                                  mask_to_rgb(gt.astype(np.int32)),
                                  mask_to_rgb(pred_up),
                                  cam_ov, sal_ov]):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")

        axes[i, 0].set_ylabel(f"mIoU = {iou:.3f}", fontsize=9,
                               rotation=0, labelpad=65, va="center")

    # Class legend
    patches = [mpatches.Patch(color=np.array(c) / 255, label=CLASSES[ci])
               for ci, c in enumerate(COLORS)]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}  ({out_path.stat().st_size // 1024} KB)")


# ── YOLOv8n-seg ───────────────────────────────────────────────────────────────
def load_yolo():
    from ultralytics import YOLO
    return YOLO(str(YOLO_PT))


def box_iou_np(a, b):
    """IoU between two xyxy boxes (numpy arrays or 4-tuples)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    au = (a[2] - a[0]) * (a[3] - a[1])
    bu = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (au + bu - inter + 1e-8)


def load_gt_boxes(lbl_path, iw, ih):
    """Load YOLO-format labels → list of (cls, x1,y1,x2,y2) absolute."""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            x1 = (cx - bw / 2) * iw
            y1 = (cy - bh / 2) * ih
            x2 = (cx + bw / 2) * iw
            y2 = (cy + bh / 2) * ih
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def yolo_image_score(yolo_model, img_path, lbl_path):
    """
    Per-image detection quality = mean best-match box-IoU over all GT objects.
    Returns 0 if no GT or no predictions.
    """
    img  = Image.open(img_path)
    iw, ih = img.size
    gt   = load_gt_boxes(lbl_path, iw, ih)
    if not gt:
        return 0.0

    res = yolo_model.predict(str(img_path), verbose=False, conf=0.25, iou=0.45)
    if res[0].boxes is None or len(res[0].boxes) == 0:
        return 0.0

    preds = res[0].boxes.xyxy.cpu().numpy()
    ious  = []
    for (_, *gt_box) in gt:
        best = max((box_iou_np(gt_box, p) for p in preds), default=0.0)
        ious.append(best)
    return float(np.mean(ious))


def rank_yolo(yolo_model, val_img_dir, val_lbl_dir, n=3):
    fnames = sorted(os.listdir(val_img_dir))
    scores = []
    print(f"  Ranking {len(fnames)} val images for YOLOv8n-seg...")
    for i, fn in enumerate(fnames):
        if i % 100 == 0:
            print(f"    {i}/{len(fnames)}", end="\r", flush=True)
        lbl_path = val_lbl_dir / (Path(fn).stem + ".txt")
        score = yolo_image_score(yolo_model, val_img_dir / fn, lbl_path)
        scores.append((score, fn))
    print()
    scores.sort(key=lambda x: x[0])
    worst = [s[1] for s in scores[:n]]
    best  = [s[1] for s in scores[-n:][::-1]]
    print(f"  Best  BoxIoU: {[round(s[0],3) for s in scores[-n:][::-1]]}")
    print(f"  Worst BoxIoU: {[round(s[0],3) for s in scores[:n]]}")
    return best, worst


def yolo_gradcam_saliency(yolo_model, img_pil):
    """
    Returns (cam_overlay, sal_overlay) at original image size.
    Score: sum of max-class confidences across all 8400 anchors.
    Target layer: backbone SPPF (layer index 9).
    """
    torch_model = yolo_model.model.cuda()
    torch_model.eval()

    img_np_orig = np.array(img_pil)
    H_orig, W_orig = img_np_orig.shape[:2]

    # Preprocess to YOLO 640×640
    img_640 = cv2.resize(img_np_orig, (640, 640))
    t = (torch.from_numpy(img_640).permute(2, 0, 1)
         .float().div(255.0).unsqueeze(0).to(DEVICE))

    def score_fn(x):
        raw = torch_model(x)
        # raw[0] is (preds, proto); preds shape (1, 41, 8400)
        preds = raw[0][0] if isinstance(raw[0], (list, tuple)) else raw[0]
        # class scores: indices [4:14], apply sigmoid
        conf = preds[0, 4:4 + len(YOLO_CLASSES), :].sigmoid()  # (10, 8400)
        return conf.max(dim=0)[0].sum()

    # --- Grad-CAM ---
    gc = GradCAM(torch_model.model[9])
    torch_model.zero_grad()
    try:
        with torch.enable_grad():
            t_g   = t.clone().detach().requires_grad_(True)
            score = score_fn(t_g)
            cam_raw = gc.cam(score)
        cam_up = resize_map(cam_raw, H_orig, W_orig)
    except Exception as e:
        print(f"    [warn] YOLO Grad-CAM failed ({e}), using uniform map")
        cam_up = np.full((H_orig, W_orig), 0.5, dtype=np.float32)
    finally:
        gc.remove()

    # --- Saliency ---
    torch_model.zero_grad()
    try:
        with torch.enable_grad():
            sal_raw = vanilla_saliency(t, score_fn)
        sal_up = resize_map(sal_raw, H_orig, W_orig)
    except Exception as e:
        print(f"    [warn] YOLO saliency failed ({e}), using uniform map")
        sal_up = np.full((H_orig, W_orig), 0.5, dtype=np.float32)

    cam_overlay = apply_heatmap(cam_up, img_np_orig)
    sal_overlay = apply_heatmap(sal_up, img_np_orig)
    return cam_overlay, sal_overlay


def plot_yolo(yolo_model, fnames, title, out_path):
    val_img_dir = SEG_ROOT / "instance" / "val" / "images"
    val_lbl_dir = SEG_ROOT / "instance" / "val" / "labels"

    n    = len(fnames)
    cols = ["Input", "GT Boxes", "Prediction", "Grad-CAM", "Saliency"]
    fig, axes = plt.subplots(n, 5, figsize=(22, 4.5 * n))
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for j, ct in enumerate(cols):
        axes[0, j].set_title(ct, fontsize=10, pad=4)

    for i, fn in enumerate(fnames):
        img_pil = Image.open(val_img_dir / fn).convert("RGB")
        img_np  = np.array(img_pil)
        H, W    = img_np.shape[:2]
        lbl_path = val_lbl_dir / (Path(fn).stem + ".txt")

        # GT boxes drawn on image
        gt_img = img_np.copy()
        for (cls, x1, y1, x2, y2) in load_gt_boxes(lbl_path, W, H):
            col = COLORS[cls + 1]  # +1 because YOLO classes skip background
            cv2.rectangle(gt_img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            label = YOLO_CLASSES[cls]
            cv2.putText(gt_img, label, (int(x1), max(int(y1) - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

        # YOLO prediction (uses built-in plot which draws masks + boxes)
        res      = yolo_model.predict(str(val_img_dir / fn), verbose=False,
                                      conf=0.25, iou=0.45)
        pred_bgr = res[0].plot()
        pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

        score = yolo_image_score(yolo_model, val_img_dir / fn, lbl_path)
        cam_ov, sal_ov = yolo_gradcam_saliency(yolo_model, img_pil)

        for j, im in enumerate([img_np, gt_img, pred_rgb, cam_ov, sal_ov]):
            axes[i, j].imshow(im)
            axes[i, j].axis("off")

        axes[i, 0].set_ylabel(f"BoxIoU = {score:.3f}", fontsize=9,
                               rotation=0, labelpad=70, va="center")

    # Class legend (YOLO 10 classes)
    patches = [mpatches.Patch(color=np.array(COLORS[ci + 1]) / 255,
                               label=YOLO_CLASSES[ci])
               for ci in range(len(YOLO_CLASSES))]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}  ({out_path.stat().st_size // 1024} KB)")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── DeepLabV3+ ────────────────────────────────────────────────────────────
    print("\n=== DeepLabV3+ ===")
    dl_model = load_deeplab()
    print(f"  Loaded {DEEPLAB_PT.name}")

    dl_val_img = SEG_ROOT / "semantic" / "val" / "images"
    dl_val_msk = SEG_ROOT / "semantic" / "val" / "masks"

    dl_best, dl_worst = rank_deeplab(dl_model, dl_val_img, dl_val_msk, n=3)

    print("  Plotting best-3...")
    plot_deeplab(dl_model, dl_best,
                 "DeepLabV3+ ResNet50 — Best 3 Predictions (keremberke val)",
                 OUT_DIR / "deeplab_best3.png")

    print("  Plotting worst-3...")
    plot_deeplab(dl_model, dl_worst,
                 "DeepLabV3+ ResNet50 — Worst 3 Predictions (keremberke val)",
                 OUT_DIR / "deeplab_worst3.png")

    # ── YOLOv8n-seg ───────────────────────────────────────────────────────────
    print("\n=== YOLOv8n-seg ===")
    yolo = load_yolo()
    print(f"  Loaded {YOLO_PT.name}")

    yo_val_img = SEG_ROOT / "instance" / "val" / "images"
    yo_val_lbl = SEG_ROOT / "instance" / "val" / "labels"

    yo_best, yo_worst = rank_yolo(yolo, yo_val_img, yo_val_lbl, n=3)

    print("  Plotting best-3...")
    plot_yolo(yolo, yo_best,
              "YOLOv8n-seg — Best 3 Predictions (keremberke val)",
              OUT_DIR / "yolo_best3.png")

    print("  Plotting worst-3...")
    plot_yolo(yolo, yo_worst,
              "YOLOv8n-seg — Worst 3 Predictions (keremberke val)",
              OUT_DIR / "yolo_worst3.png")

    print("\n=== Done ===")
    for fn in ["deeplab_best3.png", "deeplab_worst3.png",
               "yolo_best3.png",   "yolo_worst3.png"]:
        p = OUT_DIR / fn
        if p.exists():
            print(f"  {fn}  ({p.stat().st_size // 1024} KB)")
