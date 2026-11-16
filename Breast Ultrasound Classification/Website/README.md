# 🌐 Breast Ultrasound AI — Website

The trained thesis model (EfficientNet-B0 + Grad-CAM, 92.31% test accuracy) as a web app.
Upload a breast ultrasound scan → get the prediction, class probabilities, a Grad-CAM heatmap,
a **detection box** on the region that drove the decision, and a downloadable **PDF report**.

## ▶ Live demo

**https://sadhin404-breast-ultrasound-ai.static.hf.space**
(Hugging Face Space: https://huggingface.co/spaces/Sadhin404/breast-ultrasound-ai)

The live version is the **static app** below: the model runs **entirely in the visitor's
browser** (ONNX Runtime Web), so the site never sleeps and uploaded scans never leave the device.
The first analysis loads the 17 MB model once; after that it is instant.

## Features

- **Upload & analyse** any breast ultrasound image (PNG/JPG)
- **Prediction** — Normal / Benign / Malignant with confidence and per-class probability bars
- **Grad-CAM views** — original scan, heatmap and blended overlay (with a blend slider)
- **Detection box** — a box drawn on the original scan around the region that influenced the
  decision most (derived from the Grad-CAM heatmap; colour follows the predicted class)
- **PDF report** — one-page report with the result, probabilities, Grad-CAM panels,
  interpretation, recommended next step and the medical disclaimer
- **Sample scans** — three built-in examples (one per class) to try instantly
- Fully responsive, animated interface — works on phone and desktop

## What's in this folder

| Folder | What it is |
|---|---|
| `static-app/` | ⭐ The version **live now** on Hugging Face — full custom design, ONNX model runs in the browser |
| `gradio-app/` | The earlier Hugging Face version (Gradio + ZeroGPU server inference) |
| `web-app/` | The full custom web app — FastAPI backend + the same animated front end, packaged with Docker |

All versions use the **same** trained model (`../model/best_model.pth`), the same preprocessing
(Resize 224×224 + ImageNet normalisation) and the same Grad-CAM method (target layer
`model.features[-1]`) as the thesis code. The static app's ONNX model was verified to give
**identical predictions** to the PyTorch model (probability difference 0.0 on the test samples);
its Grad-CAM is computed analytically from the final classifier weights, which is mathematically
equivalent for this architecture (GAP + linear head).

## Run it yourself

**Static app (the live version)** — just serve the folder:

```bash
cd static-app
mkdir -p model && cp ../../model/best_model.pth model/   # only needed to rebuild model.onnx
python export_onnx.py        # regenerates model.onnx + cam_weights.json from the .pth
python -m http.server 8000   # then open http://localhost:8000
```

(`model.onnx` itself is a binary export of `../model/best_model.pth`, so it is not stored in git —
the two small generated files that are stored are `cam_weights.json` plus the site code.)

**Gradio version**

```bash
cd gradio-app
mkdir -p model && cp ../../model/best_model.pth model/
pip install -r requirements.txt
python app.py
```

Note: `app.py` uses the Hugging Face `spaces` package (`@spaces.GPU`) — on a plain machine
without ZeroGPU, remove the `import spaces` line and the `@spaces.GPU` decorator and it runs on CPU.

**Custom web app (FastAPI + Docker)**

```bash
cd web-app
mkdir -p model && cp ../../model/best_model.pth model/
docker build -t breast-ultrasound-ai .
docker run -p 7860:7860 breast-ultrasound-ai
# open http://localhost:7860
```

Without Docker: `pip install -r requirements.txt` then
`MODEL_PATH=model/best_model.pth uvicorn app.main:app --port 7860`.

## Tech

PyTorch · EfficientNet-B0 · Grad-CAM · ONNX Runtime Web · Gradio / FastAPI · ReportLab / jsPDF ·
Hugging Face Spaces

---

*Research tool only — not a medical diagnosis. Developed by MD. Shadhin Khan, CSE,
First Capital University of Bangladesh. Dataset: BUSI (Al-Dhabyani et al., 2020).*
