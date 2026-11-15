# 🚀 How to Run This Project — Step by Step

Follow these steps exactly and the project will run **without errors**. Total time: about **12–15 minutes** (most of it is the one-time dataset download).

---

## ✅ What you need

- A **Google account** (for Colab + Drive)
- A browser — that's it. Nothing to install on your computer.

---

## 📥 Step 1 — Download this project

On this repository page: **Code → Download ZIP**, then unzip it on your computer.
(Or `git clone https://github.com/Sadhin404/Deep-Learning-Project.git`)

## 📦 Step 2 — Get the dataset (`archive.zip`, ~204 MB)

Download **`archive.zip`** from the [**Releases page → v1.0**](https://github.com/Sadhin404/Deep-Learning-Project/releases/tag/v1.0) of this repository.

> The dataset is the **BUSI** breast ultrasound dataset. It is also available on Kaggle:
> *aryashah2k/breast-ultrasound-images-dataset* — if you download it from Kaggle instead, the zip you need is the same `archive.zip`.

## 📁 Step 3 — Put the dataset in your Google Drive (⚠️ this step decides success or failure)

1. Open **Google Drive** with your Google account.
2. In **My Drive**, create a folder named exactly:

   ```
   Cancer
   ```

   — capital **C**, no extra spaces, no other name.
3. Upload **`archive.zip`** into that `Cancer` folder. **Do not rename the zip. Do not unzip it yourself** — the notebook unzips it.

Your Drive must look like this:

```
My Drive/
└── Cancer/
    └── archive.zip
```

> ❌ **"Dataset not found" error?** It always means this step is wrong: the folder is not named exactly `Cancer`, the zip is not directly inside it, or you mounted Colab with a *different* Google account than the one that holds the folder.

## 📓 Step 4 — Open the notebook in Google Colab

From the unzipped project folder, open:

```
Breast Ultrasound Classification/Breast_Ultrasound_EfficientNetB0_GradCAM.ipynb
```

Upload this `.ipynb` file to **Google Colab** (colab.research.google.com → File → Upload notebook), or open it from GitHub inside Colab.

## ⚡ Step 5 — Select the T4 GPU (important!)

In Colab: **Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save**.

> On CPU the training takes hours. On the free T4 GPU it takes about **3.5 minutes**.

## ▶️ Step 6 — Run everything

Click **Runtime → Run all**. When Colab asks to connect Google Drive, **allow it** — and make sure it is the **same Google account** where you put the `Cancer` folder in Step 3.

Then just wait. The cells run one by one:

| Stage | What happens | Approx. time |
|---|---|---|
| Setup | Drive mounts, dataset found and unzipped | ~2 min |
| Training | 20 epochs on the T4 GPU | ~3.5 min |
| Evaluation | Test metrics are computed | ~1 min |
| Grad-CAM | Explanation heatmaps are generated | ~3 min |

## 🎉 Step 7 — See your result

The **last cell** prints the final result. You should see:

```
Test accuracy    : 92.31%
Test macro F1    : 92.46%
Test weighted F1 : 92.34%
```

The run is **fully reproducible** — the split uses seed 42, so you get the same data split and the same result every time.

All outputs (trained model, metrics, figures, Grad-CAM panels) are created inside the Colab session under `outputs/`, and the notebook's backup cell copies everything neatly into your Drive folder **`Breast_Ultrasound_Thesis/`** (folders 01–07).

---

## 🎯 Just want to try the trained model on one image? (no training)

1. Take **`model/best_model.pth`** from this repository and any single ultrasound image.
2. In Colab (or locally with PyTorch installed), run:

```bash
python predict.py --image your_image.png --checkpoint best_model.pth
```

It prints the predicted class — **Normal / Benign / Malignant** — with a confidence score.
For the explanation heatmap of a single image:

```bash
python gradcam.py --image your_image.png --checkpoint best_model.pth
```

---

## 🛠 Troubleshooting

| Problem | Cause & fix |
|---|---|
| **Dataset not found** | Step 3 is wrong — folder must be `My Drive/Cancer/archive.zip`, exact name, and Colab must be mounted with the same Google account. |
| **Training is extremely slow** | You are on CPU. Runtime → Change runtime type → **T4 GPU**, then Runtime → Restart and run all. |
| **Drive mount fails / permission popup** | Allow pop-ups and third-party cookies for Colab, and approve the Drive access prompt. |
| **A cell says a file/module is missing** | Cells were skipped or run out of order. Runtime → **Restart and run all**, top to bottom, without skipping. |
| **CUDA out of memory** | Runtime → Restart session, then run all again. Batch size 16 fits the free T4. |
| **Different accuracy** | Make sure you used the original `archive.zip` unchanged and ran all cells from the top — the seed-42 split guarantees the published split. |

---

## 📂 What is in this project folder?

```
Breast Ultrasound Classification/
├── Breast_Ultrasound_EfficientNetB0_GradCAM.ipynb   ← the notebook (run this)
├── dataset.py        ← loads the dataset, excludes masks, makes the split
├── model.py          ← EfficientNet-B0 with a 3-class head
├── train.py          ← training loop, class weights, best-checkpoint saving
├── evaluate.py       ← test-set metrics (accuracy, F1, ROC-AUC, confusion matrix)
├── predict.py        ← predict one image
├── gradcam.py        ← Grad-CAM heatmaps
├── requirements.txt  ← Python libraries needed
├── paper/            ← the full thesis paper (PDF)
├── model/            ← trained model: best_model.pth
├── results/          ← metrics, figures and Grad-CAM panels from the official run
├── dataset/          ← about the dataset + where to get it
└── HOW_TO_RUN.md     ← you are here
```

*Questions about the science? Read the paper in `paper/` — dataset details, methodology, results and limitations are all documented there.*
