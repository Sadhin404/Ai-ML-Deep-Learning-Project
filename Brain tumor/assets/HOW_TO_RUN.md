# 🚀 How to Run This Project — Step by Step

Follow these steps exactly and the project will run **without errors**. Total time: about **15–20 minutes** (most of it is the one-time dataset download and the training itself).

---

## ✅ What you need

- A **Google account** (for Colab + Drive)
- A browser — that's it. Nothing to install on your computer.

---

## 📥 Step 1 — Download this project

On this repository page: **Code → Download ZIP**, then unzip it on your computer.
(Or `git clone https://github.com/Sadhin404/Ai-ML-Deep-Learning-Project.git`)

## 📦 Step 2 — Get the dataset (`archive.zip`, ~149 MB)

Download **`archive.zip`** from Google Drive:

👉 **[Download the dataset (Google Drive)](https://drive.google.com/file/d/171SvOajLgFVQg6THsoBHnzSJofJpGlFp/view?usp=sharing)**

The downloaded file must be about **149 MB** — then it is complete.

> The dataset is the **Brain Tumor MRI Dataset** by Masoud Nickparvar (Kaggle). ⚠️ Kaggle's page now serves a *rebalanced* 7,200-image version — this project needs the **original 7,023-image version** (Testing counts 300/306/405/300, not 400 per class). The original survives at https://www.kaggle.com/dsv/2645886. Details in [DATASET.md](../dataset/DATASET.md).

## 📁 Step 3 — Put the dataset in your Google Drive (⚠️ this step decides success or failure)

1. Open **Google Drive** with your Google account.
2. In **My Drive**, create a folder named exactly:

   ```
   BrainTumor
   ```

   — capital **B** and **T**, no spaces, no other name.
3. Upload **`archive.zip`** into that `BrainTumor` folder. **Do not rename the zip. Do not unzip it yourself** — the notebook unzips it. (If you already extracted the dataset, that works too: the notebook finds the `Training`/`Testing` folders at any depth inside `BrainTumor`. Keep only **one** copy of the dataset in the folder.)

Your Drive must look like this:

```
My Drive/
└── BrainTumor/
    └── archive.zip
```

> ❌ **"Dataset not found" error?** It always means this step is wrong: the folder is not named exactly `BrainTumor`, the zip is not directly inside it, or you mounted Colab with a *different* Google account than the one that holds the folder.

## 📓 Step 4 — Open the notebook in Google Colab

From the unzipped project folder, open:

```
Brain tumor/model/BrainTumor.ipynb
```

Upload this `.ipynb` file to **Google Colab** (colab.research.google.com → File → Upload notebook), or open it from GitHub inside Colab.

## ⚡ Step 5 — Select the T4 GPU (important!)

In Colab: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**.

> On CPU the training takes hours. On the free T4 GPU it takes only minutes.

## ▶️ Step 6 — Run everything

Click **Runtime → Run all**. When Colab asks to connect Google Drive, **allow it** — and make sure it is the **same Google account** where you put the `BrainTumor` folder in Step 3.

Then just wait. The cells run one by one:

| Stage | What happens |
|---|---|
| Setup | Drive mounts, dataset found and unzipped, per-class counts printed |
| Training | 20 epochs on the T4 GPU — best checkpoint kept by validation macro F1 |
| Evaluation | The held-out test set is evaluated **once** |
| Grad-CAM | Explanation heatmaps are generated (one test image per class) |
| Saving | Everything is copied into your Drive folder **`BrainTumor/Result`** |

✅ **Check the counts when the dataset cell runs:** Training — glioma 1321, meningioma 1339, notumor 1595, pituitary 1457 · Testing — 300 / 306 / 405 / 300. If you see 1400/400 per class, you have the new rebalanced dataset version — stop and get the original (Step 2).

## 🎉 Step 7 — See your result

The evaluation cell prints the final result. The official run gave:

```
TEST ACCURACY : 99.47%
TEST MACRO F1 : 99.46%
Errors        : 7 / 1311
```

The run is **fully reproducible** — the split uses seed 42, so you get the same data split every time; small run-to-run variation of an image or two is normal for GPU training.

All outputs (trained model `best_model.pth`, `results.txt`, `confusion_matrix.png`, `training_curve.png`, `gradcam_examples.png`) are saved into your Drive folder **`BrainTumor/Result`**.

---

## 🎯 Just want to try the trained model on one image? (no training)

1. Download **`best_model.pth`** from this repository's [brain-tumor-v1.0 release](https://github.com/Sadhin404/Ai-ML-Deep-Learning-Project/releases/tag/brain-tumor-v1.0) and take any single brain MRI image.
2. In Colab (or locally with PyTorch installed), from the `assets/` folder run:

```bash
python predict.py --image your_image.jpg --checkpoint best_model.pth
```

It prints the predicted class — **Glioma / Meningioma / No Tumor / Pituitary** — with a confidence score.
For the explanation heatmap of a single image:

```bash
python gradcam.py --image your_image.jpg --checkpoint best_model.pth
```

---

## 🛠 Troubleshooting

| Problem | Cause & fix |
|---|---|
| **Dataset not found** | Step 3 is wrong — folder must be `My Drive/BrainTumor`, exact name, and Colab must be mounted with the same Google account. |
| **Wrong class counts (1400 / 400)** | You downloaded Kaggle's new rebalanced version. Get the original from https://www.kaggle.com/dsv/2645886 or use the Drive link in Step 2. |
| **Training is extremely slow** | You are on CPU. Runtime → Change runtime type → **T4 GPU**, then Runtime → Restart and run all. |
| **Drive mount fails / permission popup** | Allow pop-ups and third-party cookies for Colab, and approve the Drive access prompt. |
| **A cell says a file/module is missing** | Cells were skipped or run out of order. Runtime → **Restart and run all**, top to bottom, without skipping. |
| **CUDA out of memory** | Runtime → Restart session, then run all again. Batch size 32 fits the free T4. |
| **Different accuracy** | Make sure you used the original `archive.zip` unchanged and ran all cells from the top — the seed-42 split guarantees the published split. |

---

## 📂 What is in this project folder?

```
Brain tumor/
├── assets/           ← all the code + this guide
│   ├── dataset.py        ← finds the dataset, makes the seed-42 train/val split
│   ├── model.py          ← DenseNet-201 with a 4-class head
│   ├── train.py          ← training loop, best-checkpoint saving (validation F1)
│   ├── evaluate.py       ← test-set metrics (accuracy, F1, confusion matrix)
│   ├── predict.py        ← predict one image
│   ├── gradcam.py        ← Grad-CAM heatmaps
│   ├── requirements.txt  ← Python libraries needed
│   └── HOW_TO_RUN.md     ← you are here
├── model/            ← the Colab notebook (run this) — the checkpoint is in the release
├── paper/            ← the full project paper (PDF)
├── slide/            ← the presentation (PPTX)
├── results/          ← metrics and figures from the official run
├── dataset/          ← about the dataset + where to get it
└── Website/          ← the live web app (static, runs in the browser)
```

*Questions about the science? Read the paper in `../paper/` — dataset details, methodology, results and limitations are all documented there.*
