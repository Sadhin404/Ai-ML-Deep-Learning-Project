# Dataset — Chest X-Ray Images (Pneumonia)

## Source

**Kermany et al. (2018)** — chest X-rays of paediatric patients (ages 1–5)
from the Guangzhou Women and Children's Medical Center, released with the
paper *"Identifying Medical Diagnoses and Treatable Diseases by Image-Based
Deep Learning"* (Cell, 2018).

📥 **Official download (Kaggle):**
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset is **not included in this repository** — download it from the
official Kaggle page above. The Kaggle export downloads as `archive.zip`
(~2.46 GB; the size is inflated by duplicate copies inside the zip — the
unique images are the 5,856 counted below).

## Contents

Two classes of JPEG chest X-rays:

| Class | Meaning |
| --- | --- |
| `NORMAL` | Healthy chest, lungs clear |
| `PNEUMONIA` | Opacities consistent with pneumonia (bacterial and viral cases are in one class) |

Official folder split, counted from the actual files:

| Split | Normal | Pneumonia | Total |
| --- | --- | --- | --- |
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

> ⚠️ Kaggle's page often quotes **5,863** images. The folders themselves
> contain **5,856** files. Every count in this project comes from the files,
> not the page.

## How this project uses it

- The official validation split (16 images) is too small to select
  checkpoints with, so **train + validation are pooled (5,232 images)** and
  re-split once — **seed 42, stratified 88/12** → **4,604 train /
  628 validation** (Normal 1,187 / 162, Pneumonia 3,417 / 466).
- The **official test set (624 images) is never touched during training or
  tuning** and is evaluated exactly once per model at the end.
- Preprocessing: training images are resized to 256, randomly cropped to
  224 (scale 0.85–1.0), flipped, rotated ±15° and jittered
  (brightness/contrast 0.1); validation and test images are resized straight
  to 224. All images are normalised with the ImageNet mean/std.
- The loss is class-weighted (≈ 1.94 Normal / 0.67 Pneumonia) to counter the
  pool's imbalance.

## Expected folder layout

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The notebook also accepts the raw `archive.zip` (it extracts and finds the
folders itself), or a split zip uploaded as `chest_xray.zip.part-aa`,
`.part-ab`, … — see [`../assets/HOW_TO_RUN.md`](../assets/HOW_TO_RUN.md).

## Licence / credit

The dataset belongs to its original authors (Kermany et al.). If you use it,
cite the original Cell paper and the Kaggle dataset page. This project uses
it for research and education only.
