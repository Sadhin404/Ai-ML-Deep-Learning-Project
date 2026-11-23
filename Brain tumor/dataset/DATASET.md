# 📦 The Dataset — Brain Tumor MRI (Nickparvar)

## Where the dataset file lives

The dataset zip (**`archive.zip`**, ~149 MB) is **not stored in this repository** — GitHub does not allow single files larger than 100 MB in a repo. You can get it in either of these ways:

1. **From Google Drive (recommended):** [Download `archive.zip`](https://drive.google.com/file/d/171SvOajLgFVQg6THsoBHnzSJofJpGlFp/view?usp=sharing) — one file, ~149 MB.
2. **From Kaggle:** *masoudnickparvar/brain-tumor-mri-dataset* — ⚠️ the page's **current** version is a rebalanced 7,200-image edition. This project uses the **original version**, which is still available at the version link: https://www.kaggle.com/dsv/2645886 — download from there so the counts below match exactly.

Then place it in your Google Drive as **`My Drive/BrainTumor/archive.zip`** — see **[HOW_TO_RUN.md](../assets/HOW_TO_RUN.md)**.

## What is inside the zip

```
Training/
├── glioma/       1,321 images
├── meningioma/   1,339 images
├── notumor/      1,595 images
└── pituitary/    1,457 images
Testing/
├── glioma/         300 images
├── meningioma/     306 images
├── notumor/        405 images
└── pituitary/      300 images
```

- **7,023 real brain MRI images** in total, in four classes — **Glioma**, **Meningioma**, **No Tumor** and **Pituitary**.
- Images are JPG, variable size; the model resizes them to 224×224.
- ✅ **Quick check:** if your Testing counts are 300 / 306 / 405 / 300, you have the original version. If they are 400 per class, you downloaded the new rebalanced version — get the original from the version link above instead.

## The split used in this project

The dataset's own **Testing** folder (1,311 images) is the held-out test set — it is evaluated **once**, at the very end. The **Training** folder is split, stratified with seed **42**:

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---|---|---|---|---|
| Train | 1,162 | 1,178 | 1,404 | 1,282 | 5,026 |
| Validation | 159 | 161 | 191 | 175 | 686 |
| Test (Testing folder) | 300 | 306 | 405 | 300 | 1,311 |

The notebook regenerates this exact split from the seed, so the experiment is fully reproducible. The test set was never used for training or model selection.

## Source & citation

The dataset was published on Kaggle by **Masoud Nickparvar**:

> Nickparvar, M. (2021). *Brain Tumor MRI Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Please cite the original author if you use the dataset. It is shared for academic/research use.
