# The Dataset — Rice Leaf Disease Image

> The dataset itself is **not** included in this repository. This page documents what was used, where to get it, and exactly how it was split and preprocessed.

## Source

**Rice Leaf Disease Image** — originally published on Kaggle by Nirmal Sankalana:

📥 **https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image**

The notebook loads the identical dataset from its Hugging Face mirror —
[`Project-AgML/rice_leaf_disease_classification_india`](https://huggingface.co/datasets/Project-AgML/rice_leaf_disease_classification_india) —
so the run works in Google Colab without a Kaggle download. All counts below were verified against the actual files.

## Contents

**5,932 images** of rice leaves, roughly 300 × 300 pixels, in four classes:

| Class | Cause | Images |
| --- | --- | --- |
| Bacterial Blight | Bacterium — *Xanthomonas oryzae* pv. *oryzae* | 1,584 |
| Blast | Fungus — *Magnaporthe oryzae* | 1,440 |
| Brown Spot | Fungus — *Bipolaris oryzae* | 1,600 |
| Tungro | Virus (RTBV/RTSV), spread by the green leafhopper | 1,308 |
| **Total** | | **5,932** |

The classes are close to balanced (largest : smallest ≈ 1.22 : 1), so no class weighting was applied. The images are comparatively clean — the leaf fills the frame and backgrounds are simple — which is one reason the measured scores are as high as they are; the paper discusses this openly.

## The split used in this project

The dataset ships without an official split, so one was made **once** and frozen:

- **Seed 42, stratified**, in two stages: first 70% / 30%, then the 30% halved.
- **Train 4,152 · Validation 890 · Test 890** (70/15/15).
- Test support per class: Bacterial Blight 237 · Blast 216 · Brown Spot 240 · Tungro 197.
- The **test set is evaluated exactly once**, after training and checkpoint selection are finished. Re-running the notebook with seed 42 reproduces the identical split.

## Preprocessing

| | Training | Validation / Test |
| --- | --- | --- |
| Resize | 256 × 256 | 256 × 256 |
| Crop | Random resized crop 224 (scale 0.8–1.0) | Centre crop 224 |
| Flip | Random horizontal | — |
| Rotation | ±15° | — |
| Colour jitter | Brightness 0.2, contrast 0.2 | — |
| Normalise | ImageNet mean/std | ImageNet mean/std |

## Credit

All rights to the images belong to the dataset's original creators. If you use the dataset, cite the Kaggle page above. This project uses it for academic research only.
