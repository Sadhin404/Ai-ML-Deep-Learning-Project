# Website — Rice Leaf Disease Classification

The live web app for this project. Upload a photo of a rice leaf and the app names the disease, shows the Grad-CAM evidence, and lists what to do next.

## ▶ Try it

- **Live site:** https://fcub-cse-rice-leaf-disease.static.hf.space
- **Short link:** https://is.gd/RiceLeafDisease
- **Hugging Face Space:** https://huggingface.co/spaces/FCUB-CSE/rice-leaf-disease

## What it does

1. **Upload** a rice leaf photo (or tap one of the four sample leaves).
2. **Pick a model** — EfficientNet-B0 or MobileNetV2; both run entirely in the browser.
3. **Read the result** — the **disease name**, a **Grad-CAM heatmap** over the leaf showing which region drove the decision, and a **"You should take these steps"** list with three practical management steps for that disease.

Deliberately, the site shows **no accuracy or confidence numbers** — a name, the evidence, and the next steps are what a farmer needs; a percentage could be misread as a promise. There is no sign-up, no server, and uploaded photos never leave the device.

## How it works

- **Static hosting** on Hugging Face Spaces — there is no backend at all.
- Both models were exported to **dual-output ONNX** (`logits` + final `features`) and run with **ONNX Runtime Web**. The Grad-CAM map is computed analytically from the features and the classifier weights (`cam_weights_*.json`), exactly as in the notebook.
- Before deployment, both ONNX models were verified against PyTorch on real images — identical predictions.
- The interface is intentionally minimal: one upload, one model choice, one answer.

## The code

[`static-app/`](static-app/) holds the exact files served by the live site — `index.html`, `app.js`, `style.css`, both ONNX models, both CAM weight files and the four sample leaves. Unlike larger projects, both models are small enough (16.0 MB and 8.9 MB) to live here in full, so this copy of the site runs as-is: serve the folder with any static file server and open `index.html`.

## A note on use

This is an agricultural decision-support aid, not a substitute for an agricultural extension officer or plant pathologist. Serious or spreading infections should always be confirmed by a local expert before treatment.
