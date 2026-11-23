# 🌐 Brain Tumor MRI AI — Website

The trained project model (DenseNet-201 + Grad-CAM, 99.47% test accuracy) as a web app.
Upload a brain MRI scan → get the prediction, class probabilities, a Grad-CAM heatmap,
a **focus box** on the region that drove the decision, and a downloadable **PDF report**.

## ▶ Live demo

**https://fcub-cse-brain-tumor-mri-ai.static.hf.space**
(Hugging Face Space: https://huggingface.co/spaces/FCUB-CSE/brain-tumor-mri-ai)

**Short link: https://tinyurl.com/BrainTumorMRI**

The live version is the **static app** below: the model runs **entirely in the visitor's
browser** (ONNX Runtime Web), so the site never sleeps and uploaded scans never leave the device.
The first analysis loads the ~37 MB model once; after that it is fast.

## Features

- **Upload & analyse** any brain MRI image (PNG/JPG)
- **MRI check** — before analysing, the site verifies the upload really looks like a brain MRI scan (greyscale, scan texture); photos, screenshots and documents are refused with an explanation
- **Prediction** — Glioma / Meningioma / No Tumor / Pituitary with confidence and per-class probability bars
- **Grad-CAM views** — original scan, heatmap and blended overlay (with a blend slider)
- **Focus box** — a box drawn on the scan around the region that influenced the decision most (derived from the Grad-CAM heatmap; colour follows the predicted class)
- **PDF report** — one-page report with the result, probabilities, Grad-CAM panels, interpretation, recommended next step and the medical disclaimer
- **Sample scans** — four built-in examples (one per class) to try instantly
- Minimal, light, fully responsive design — works on phone and desktop

## What's in this folder

| Folder | What it is |
|---|---|
| `static-app/` | ⭐ The version **live now** on Hugging Face — minimal light design, ONNX model runs in the browser |

The static app uses the **same** trained model as the notebook (`../model/BrainTumor.ipynb`),
the same preprocessing (resize 224×224 + ImageNet normalisation) and the same Grad-CAM
method (target layer: the last dense block). Its ONNX export was verified to give
**identical predictions** to the PyTorch model (100% argmax agreement; maximum probability
difference 0.0005), and its in-browser Grad-CAM — computed analytically from the final
classifier weights — was verified against true gradient-based Grad-CAM (correlation 1.0000),
which is mathematically exact for this architecture (GAP + linear head).

## Run it yourself

**Static app (the live version)** — just serve the folder:

```bash
cd static-app
mkdir -p model && cp <release>/best_model.pth model/   # only needed to rebuild model.onnx
python export_onnx.py        # regenerates model.onnx + cam_weights.json from the .pth
python -m http.server 8000   # then open http://localhost:8000
```

(`model.onnx` itself is a binary export of the trained checkpoint, so it is not stored in
git — the checkpoint is attached to the
[brain-tumor-v1.0 release](https://github.com/Sadhin404/Ai-ML-Deep-Learning-Project/releases/tag/brain-tumor-v1.0).
The small generated file that *is* stored is `cam_weights.json`, plus the site code and the
four sample scans.)

## Tech

PyTorch · DenseNet-201 · Grad-CAM · ONNX Runtime Web · jsPDF · Hugging Face Spaces

---

*Research tool only — not a medical diagnosis. CSE, First Capital University of Bangladesh.
Dataset: Brain Tumor MRI Dataset (Nickparvar, Kaggle).*
