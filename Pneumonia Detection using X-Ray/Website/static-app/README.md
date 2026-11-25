# Pneumonia X-Ray AI — static web app

The exact code running live at
**https://fcub-cse-pneumonia-xray-ai.static.hf.space**
(short link: https://is.gd/PneumoniaXrayAI).

| File | Role |
| --- | --- |
| `index.html` | Page structure |
| `style.css` | Styling (minimal light theme) |
| `app.js` | Upload handling, the X-ray validity gate, ONNX inference, analytic Grad-CAM, focus box, PDF report |
| `cam_weights.json` | Classifier weights used to build the Grad-CAM analytically |
| `samples/` | Four real test-set scans (2 Normal, 2 Pneumonia) offered as one-tap samples |

**Missing on purpose:** `model.onnx` (28 MB) — too large for the git tree.
The live Space serves it alongside these files. To run the app yourself,
export it from the `best_DenseNet121.pth` checkpoint (attached to the
[pneumonia-v1.0 release](https://github.com/Sadhin404/Ai-ML-Deep-Learning-Project/releases/tag/pneumonia-v1.0))
as a dual-output model (logits + final convolutional features, opset 14)
and place it next to `index.html`.

⚠️ Research & education tool — not a medical diagnosis.
