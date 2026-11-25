# Website — Pneumonia X-Ray AI

The live web app for this project: the winning **DenseNet121** model running
**100% in the visitor's browser** — no server, no upload, the image never
leaves the device.

**▶ Live demo: [fcub-cse-pneumonia-xray-ai.static.hf.space](https://fcub-cse-pneumonia-xray-ai.static.hf.space)**
**▶ Short link: [is.gd/PneumoniaXrayAI](https://is.gd/PneumoniaXrayAI)**

## What it does

1. **Upload a chest X-ray** (or tap one of the built-in sample scans).
2. **X-ray check first** — a validity gate inspects the image (colour,
   texture and flat-area statistics calibrated on real Kermany scans) and
   refuses screenshots, photos and other non-X-ray images with a polite
   message. No prediction is produced for a refused image.
3. **Prediction** — the result shows the disease name: **Normal** or
   **Pneumonia**.
4. **Explanation** — the viewer shows the original scan and the **Grad-CAM
   overlay** with a focus box on the region that drove the decision.
5. **PDF report** — a downloadable report with the prediction, the class
   probabilities, the original scan and the overlay, and a suggested next
   step.

## How it works

- The PyTorch checkpoint was exported to a **dual-output ONNX model**
  (logits + final convolutional features), so the Grad-CAM is computed
  **analytically** in the browser from the classifier weights — the same
  explanation method as the notebook, not an approximation.
- Inference uses **ONNX Runtime Web** (WebAssembly). The exported model was
  verified against PyTorch before deployment: it reproduces the official
  result **exactly** — 95.35% accuracy and the identical confusion matrix
  [[210, 24], [5, 385]] on the 624-image test set.
- Hosting is a **static Hugging Face Space** under the FCUB-CSE
  organisation — there is no backend to maintain or wake up.

## Folders

| Path | What it is |
| --- | --- |
| `static-app/` | The exact code running on the live Space: `index.html`, `style.css`, `app.js`, `cam_weights.json`, `samples/` |

The model file itself (`model.onnx`, 28 MB) is **not in this repository** —
it is served by the live Space. It is exported from the
`best_DenseNet121.pth` checkpoint attached to the
[pneumonia-v1.0 release](https://github.com/Sadhin404/Ai-ML-Deep-Learning-Project/releases/tag/pneumonia-v1.0).

## ⚠️ Disclaimer

Research and education tool — **not a medical diagnosis**. Any real scan
must be read by a qualified clinician.
