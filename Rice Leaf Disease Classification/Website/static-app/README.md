# Static Web App — Rice Leaf Disease Classification

The exact files served by the live site:
**https://fcub-cse-rice-leaf-disease.static.hf.space** (short link: https://is.gd/RiceLeafDisease)

| File | Role |
| --- | --- |
| `index.html` | Page structure — upload, model choice, result, steps |
| `app.js` | All logic: preprocessing, ONNX inference, analytic Grad-CAM, per-disease steps |
| `style.css` | The minimal white interface |
| `model_efficientnet_b0.onnx` | EfficientNet-B0, dual-output (logits + features), 16.0 MB |
| `model_mobilenet_v2.onnx` | MobileNetV2, dual-output (logits + features), 8.9 MB |
| `cam_weights_efficientnet_b0.json` | Classifier weights for the analytic Grad-CAM (EfficientNet-B0) |
| `cam_weights_mobilenet_v2.json` | Classifier weights for the analytic Grad-CAM (MobileNetV2) |
| `samples/` | Four sample leaves, one per class, from the held-out test set |

Both models are included in full, so this copy runs as-is: serve the folder
with any static file server (for example `python -m http.server`) and open
`index.html`. Inference runs 100% in the browser via ONNX Runtime Web —
uploaded photos never leave the device.
