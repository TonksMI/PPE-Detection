"""
Person Detection Dataset Preparation
=====================================
Builds a YOLO-format person detection dataset from two sources:

1. Jomarkow Hard Hat Workers (1000 images):
   - Head and helmet bounding boxes are expanded to full-body person boxes
   - Head height * 7.5 = approximate body height
   - Explicit class-2 (person) boxes used where available

2. INRIA Person Dataset (downloaded from HuggingFace):
   - 300 pedestrian crops where the entire image IS a person
   - Box = full image (with 5% margin)

Output: YOLO format with class 0 = person
        person_detection/train/ and person_detection/val/
"""

import os, glob, shutil, random, cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOMARK_I  = os.path.join(BASE, 'datasets/jomarkow/images')
JOMARK_L  = os.path.join(BASE, 'datasets/jomarkow/labels')
OUT_ROOT  = os.path.join(BASE, 'datasets/person_detection')

# ── Download INRIA crops from HuggingFace ────────────────────────
def download_inria(n=300):
    """Download n INRIA pedestrian crop images from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Install: pip install huggingface_hub")
        return []

    save_dir = os.path.join(BASE, 'datasets/inria_person/pedestrians')
    os.makedirs(save_dir, exist_ok=True)

    files = list(list_repo_files('marcelarosalesj/inria-person', repo_type='dataset'))
    ped_files = [f for f in files if '/pedestrians/' in f][:n]
    print(f'Downloading {len(ped_files)} INRIA crops...', flush=True)

    def _dl(fname):
        try:
            local = hf_hub_download(
                repo_id='marcelarosalesj/inria-person',
                filename=fname, repo_type='dataset',
                local_dir=os.path.join(BASE, 'datasets/inria_person_raw')
            )
            out = os.path.join(save_dir, os.path.basename(fname))
            shutil.copy(local, out)
            return True
        except:
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        for fut in as_completed({ex.submit(_dl, f): f for f in ped_files}):
            if fut.result(): ok += 1
    print(f'Downloaded {ok}/{len(ped_files)}', flush=True)
    return glob.glob(f'{save_dir}/*.png') + glob.glob(f'{save_dir}/*.jpg')


# ── Build dataset ────────────────────────────────────────────────
def build_dataset():
    for split in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        os.makedirs(f'{OUT_ROOT}/{split}', exist_ok=True)

    random.seed(42)
    count = {'jomarkow': 0, 'inria': 0, 'boxes': 0}

    # ── 1. Jomarkow: expand head/helmet boxes → person boxes ──────
    lbl_files = sorted(glob.glob(f'{JOMARK_L}/*.txt'))
    random.shuffle(lbl_files)

    for lf in lbl_files:
        stem = Path(lf).stem
        img_path = f'{JOMARK_I}/{stem}.png'
        if not os.path.exists(img_path):
            img_path = f'{JOMARK_I}/{stem}.jpg'
        if not os.path.exists(img_path):
            continue

        person_boxes = []
        for line in open(lf):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            if cid == 2:
                # Explicit person box
                person_boxes.append((cx, cy, bw, bh))
            elif cid in (0, 1):
                # Head/helmet → expand to full body
                # Heuristic: head ≈ 1/7.5 body height, shift centre down
                body_h = min(bh * 7.5, 1.0)
                new_cy = cy + bh * 3.0
                new_bw = min(bw * 2.5, 1.0)
                new_cy = min(max(new_cy, body_h / 2), 1.0 - body_h / 2)

                is_dup = any(
                    abs(cx - pb[0]) < 0.05 and abs(new_cy - pb[1]) < 0.1
                    for pb in person_boxes
                )
                if not is_dup:
                    person_boxes.append((cx, new_cy, new_bw, min(body_h, 1.0)))

        if not person_boxes:
            continue

        split = 'train' if count['jomarkow'] < 850 else 'val'
        shutil.copy(img_path, f'{OUT_ROOT}/{split}/images/{stem}.png')
        with open(f'{OUT_ROOT}/{split}/labels/{stem}.txt', 'w') as f:
            for (cx, cy, bw, bh) in person_boxes:
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                bw = min(max(bw, 0.01), 1.0)
                bh = min(max(bh, 0.01), 1.0)
                f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

        count['jomarkow'] += 1
        count['boxes'] += len(person_boxes)

    print(f'Jomarkow: {count["jomarkow"]} images, {count["boxes"]} boxes', flush=True)

    # ── 2. INRIA: each crop IS a person (full-image bbox) ─────────
    inria_save = os.path.join(BASE, 'datasets/inria_person/pedestrians')
    inria_imgs = glob.glob(f'{inria_save}/*.png') + glob.glob(f'{inria_save}/*.jpg')

    if not inria_imgs:
        inria_imgs = download_inria(300)

    random.shuffle(inria_imgs)
    for i, img_path in enumerate(inria_imgs):
        stem = f'inria_{Path(img_path).stem}'
        split = 'train' if i < int(len(inria_imgs) * 0.85) else 'val'
        shutil.copy(img_path, f'{OUT_ROOT}/{split}/images/{stem}.png')
        with open(f'{OUT_ROOT}/{split}/labels/{stem}.txt', 'w') as f:
            f.write('0 0.500000 0.500000 0.900000 0.900000\n')
        count['inria'] += 1

    print(f'INRIA: {count["inria"]} images', flush=True)

    train_n = len(glob.glob(f'{OUT_ROOT}/train/images/*'))
    val_n   = len(glob.glob(f'{OUT_ROOT}/val/images/*'))
    print(f'\nDataset: {train_n} train / {val_n} val')
    print(f'Config:  {OUT_ROOT}/person_detect.yaml')


if __name__ == '__main__':
    build_dataset()
