# 📦 The Dataset — BUSI (Breast Ultrasound Images)

## Where the dataset file lives

The dataset zip (**`archive.zip`**, ~204 MB) is **not stored as a regular file in this repository** — GitHub does not allow single files larger than 100 MB in a repo. You can get it in either of these ways:

1. **From this repo's Releases (recommended):** [Releases → v1.0 → `archive.zip`](https://github.com/Sadhin404/Deep-Learning-Project/releases/tag/v1.0)
2. **From Kaggle:** *aryashah2k/breast-ultrasound-images-dataset*

Then place it in your Google Drive as **`My Drive/Cancer/archive.zip`** — see **[HOW_TO_RUN.md](../HOW_TO_RUN.md)**.

## What is inside the zip

```
Dataset_BUSI_with_GT/
├── normal/       133 images  (+ 133 masks)
├── benign/       437 images  (+ 454 masks)
└── malignant/    210 images  (+ 211 masks)
```

- **780 real ultrasound images** in total — Normal **133**, Benign **437**, Malignant **210**
- Every image also has a **ground-truth mask** file (`name_mask.png`) drawn by radiologists, showing the lesion boundary. Masks are for *segmentation* — this project is *classification*, so the code **always excludes `_mask` files** from training and testing.
- Images are PNG, variable size (about 190×310 up to 1048×719 px); the model resizes them to 224×224.
- One known quirk of the published data: `benign (433).png` and `malignant (145).png` are byte-identical images with different labels. It is kept as published and documented in the thesis.

## The split used in this project

Stratified, seed **42**, 70 / 15 / 15:

| Split | Normal | Benign | Malignant | Total |
|---|---|---|---|---|
| Train | 93 | 306 | 147 | 546 |
| Validation | 20 | 65 | 32 | 117 |
| Test | 20 | 66 | 31 | 117 |

The exact assignment is saved in [`results/split_indices.json`](../results/split_indices.json), so the experiment is fully reproducible. The test set was never used for training or model selection.

## Source & citation

The BUSI dataset was collected at Baheya Hospital, Egypt, from 600 female patients aged 25–75, and published as:

> Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). *Dataset of breast ultrasound images.* Data in Brief, 28, 104863. https://doi.org/10.1016/j.dib.2019.104863

Please cite the original authors if you use the dataset. It is shared for academic/research use.
