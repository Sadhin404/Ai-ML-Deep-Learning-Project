# How to Run — Rice Leaf Disease Classification

Everything runs from **one notebook**: [`model/Rice_Leaf_Disease_EfficientNetB0_vs_MobileNetV2_T4.ipynb`](../model/Rice_Leaf_Disease_EfficientNetB0_vs_MobileNetV2_T4.ipynb).
It trains **EfficientNet-B0** and **MobileNetV2** under one identical protocol and produces every number and figure in the paper. The expected total runtime on a free Colab T4 GPU is well under an hour.

---

## 1. Open the notebook in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in.
2. **File → Upload notebook** and choose `Rice_Leaf_Disease_EfficientNetB0_vs_MobileNetV2_T4.ipynb` from the `model/` folder — or open it straight from this repository via **Open in Colab**.
3. Set the runtime to GPU: **Runtime → Change runtime type → Hardware accelerator: T4 GPU**.

## 2. Run all cells

**Runtime → Run all.** The notebook, in order:

1. Installs/imports what it needs and checks the GPU.
2. **Downloads the dataset automatically** from its Hugging Face mirror (`Project-AgML/rice_leaf_disease_classification_india`) — no Kaggle account or manual download is needed. (The original dataset is [on Kaggle](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image); see [`dataset/DATASET.md`](../dataset/DATASET.md).)
3. Splits the 5,932 images once — **seed 42, stratified 70/15/15 → 4,152 train / 890 validation / 890 test**.
4. Trains **EfficientNet-B0** (12 epochs), keeping the best-validation-accuracy checkpoint.
5. Trains **MobileNetV2** with the identical recipe, same rule.
6. Evaluates each model on the 890-image test set **exactly once** and saves `results.json`, the two classification reports, the confusion matrices, training curves, ROC curves, Grad-CAM figures and the two checkpoints (`best_efficientnet_b0.pth`, `best_mobilenet_v2.pth`).

> ⚠️ The order matters: the test set is touched only in the final evaluation step. Do not re-run the test cells to "try again" — the reported result is the first and only evaluation.

## 3. Expected results

| Model | Test accuracy | Macro F1 | Errors |
| --- | --- | --- | --- |
| EfficientNet-B0 | 100% (890/890) | 1.000 | 0 |
| MobileNetV2 | 100% (890/890) | 1.000 | 0 |

Both models reached 100% validation accuracy early (EfficientNet-B0 at epoch 2, MobileNetV2 at epoch 3). The training loss settling near **0.36** is expected — label smoothing (0.1) sets that floor; it is not a bug. Small run-to-run variation is normal for GPU training, but on this benchmark both models are expected at or extremely near the ceiling.

## 4. The trained models

The checkpoints from the official run are attached to the
[**rice-leaf-v1.0 release**](https://github.com/Sadhin404/Ai-ML-Deep-Learning-Project/releases/tag/rice-leaf-v1.0)
(`best_efficientnet_b0.pth` 16.4 MB · `best_mobilenet_v2.pth` 9.2 MB) — use those if you only need the trained models, without retraining.

## 5. Running locally instead

```bash
pip install -r requirements.txt
jupyter notebook model/Rice_Leaf_Disease_EfficientNetB0_vs_MobileNetV2_T4.ipynb
```

A GPU is strongly recommended; on CPU the two trainings take many hours.

## Common problems

| Problem | Fix |
| --- | --- |
| `CUDA out of memory` | Restart the runtime and run again; do not raise the batch size (32 is part of the protocol). |
| Dataset download is slow or interrupted | Re-run the download cell — the Hugging Face cache resumes where it stopped. |
| `No GPU available` | Runtime → Change runtime type → T4 GPU; free-tier GPUs are sometimes busy — try again later. |
| Loss plateaus around 0.36 | That is the label-smoothing floor. Expected — not an error. |
