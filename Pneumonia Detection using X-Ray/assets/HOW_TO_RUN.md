# How to Run — Pneumonia Detection using X-Ray

Everything runs from one notebook: [`../model/Pneumonia.ipynb`](../model/Pneumonia.ipynb).
It trains **DenseNet121** and **ResNet50** under one identical protocol,
evaluates both **once** on the official test set, runs the McNemar test and
builds the Grad-CAM comparison. Follow the steps exactly and you will
reproduce the published run.

## 1. What you need

- A Google account (for Colab + Drive).
- The dataset zip — download `archive.zip` from the official Kaggle page:
  [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
  (see [`../dataset/DATASET.md`](../dataset/DATASET.md)). The dataset is
  **not** in this repository.
- About 2 hours of Colab GPU time for both models.

## 2. Prepare Google Drive

1. In Google Drive, create a folder named **`Pneumonia`** (a trailing space
   in the name is fine — the notebook tolerates it).
2. Upload `archive.zip` into that folder. The zip is ~2.46 GB because it
   contains duplicate copies inside; that is normal for this Kaggle export.
   - If your upload splits the zip into parts (`chest_xray.zip.part-aa`,
     `.part-ab`, …), put the parts in the same folder — the notebook joins
     them itself.
   - If you already extracted the dataset, that works too: the notebook
     finds `chest_xray/train`, `chest_xray/val` and `chest_xray/test`
     folders, or a single `chest_xray.zip`, automatically.

## 3. Run the notebook

1. Upload `Pneumonia.ipynb` to Colab (or open it from Drive).
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run the cells **top to bottom**. The notebook will:
   - mount Drive and locate the `Pneumonia` folder and the dataset by
     itself;
   - pool the official train + validation folders (5,232 images) and make
     the fixed **seed-42 stratified 88/12 split** → 4,604 train /
     628 validation;
   - train **DenseNet121** (SGD, lr 0.001, momentum 0.9, weight decay 1e-4,
     cosine schedule, 25 epochs, batch 32), keeping the checkpoint with the
     best validation macro F1;
   - train **ResNet50** (Adam, lr 0.001, step schedule halved every
     10 epochs, 30 epochs, batch 16), same checkpoint rule;
   - evaluate each model **exactly once** on the official 624-image test
     set;
   - run the **McNemar test** on the paired predictions and build the
     **Grad-CAM** comparison figure.

## 4. What you get

Everything is saved into a **`Result`** folder inside the Drive `Pneumonia`
folder:

| Output | Contents |
| --- | --- |
| `results.txt` | Accuracy, macro F1, sensitivity, specificity, per-class precision/recall/F1, ROC-AUC, PR-AUC, confusion matrices, McNemar result |
| `best_DenseNet121.pth` | Best DenseNet121 checkpoint (by validation macro F1) |
| `best_ResNet50.pth` | Best ResNet50 checkpoint |
| `confusion_matrices.png` + per-model matrices | Test-set confusion matrices |
| `training_curves.png` | Loss and validation macro F1 per epoch, both models |
| `roc_pr_curves.png` | ROC and precision–recall curves |
| `metrics_comparison.png` | Head-to-head metric chart |
| `gradcam_comparison.png` | Grad-CAM overlays for both models on the same films |

## 5. Expected result

| Model | Test accuracy | Macro F1 |
| --- | --- | --- |
| DenseNet121 | **95.35%** | **94.96%** |
| ResNet50 | 93.75% | 93.10% |

McNemar: b = 15, c = 5, χ² = 4.05, **p = 0.0442**.

The split is regenerated from seed 42, so your split is identical to the
official run's. GPU training is not perfectly deterministic, so a re-run can
differ by an image or two — that is normal and is not a setup error.

## 6. Common problems

- **"Dataset not found"** — the Drive folder must be named `Pneumonia` and
  contain either the extracted `chest_xray` folders, `archive.zip`, or the
  `chest_xray.zip.part-*` files. Check the folder is in **My Drive**, not in
  a shared drive the notebook cannot see.
- **Out of memory** — make sure the runtime really is a GPU (T4). On CPU the
  run is impractically slow; on smaller GPUs, restart the runtime and run
  the cells again from the top.
- **Different numbers** — if your dataset copy is the rebalanced or a
  re-split version from another source, counts will differ. Use the official
  Kaggle download; the folders must total **5,856** images (Normal 1,583 /
  Pneumonia 4,273).
