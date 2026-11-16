"""Regenerate model.onnx + cam_weights.json for the static web app.

The static app runs the model in the browser (ONNX Runtime Web). It needs:
  - model.onnx      — the trained EfficientNet-B0 exported with TWO outputs:
                      the class logits and the last convolution block's
                      activations (needed for Grad-CAM in the browser)
  - cam_weights.json — the final classifier's weight matrix. For this
                      architecture (global average pooling + one linear layer)
                      the Grad-CAM channel weights are exactly these classifier
                      weights, so the browser can compute an exact Grad-CAM
                      without backpropagation.

Usage:
    mkdir -p model && cp ../../model/best_model.pth model/
    pip install torch torchvision onnx onnxruntime pillow numpy
    python export_onnx.py
"""
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

CLASSES = ["normal", "benign", "malignant"]
CKPT = "model/best_model.pth"


def build_model(num_classes=3):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class Wrap(nn.Module):
    """Return (logits, last-conv activations) from one forward pass."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        acts = self.m.features(x)  # (1, 1280, 7, 7)
        logits = self.m(x)         # (1, 3)
        return logits, acts


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model(len(ckpt.get("classes", CLASSES)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.onnx.export(
        Wrap(model).eval(),
        torch.zeros(1, 3, 224, 224),
        "model.onnx",
        input_names=["input"],
        output_names=["logits", "acts"],
        opset_version=17,
    )
    # The default (dynamo) exporter writes weights to a sidecar .data file —
    # inline them so model.onnx is one self-contained file for the browser.
    import onnx
    _m = onnx.load("model.onnx")
    onnx.save_model(_m, "model.onnx", save_as_external_data=False)
    print("exported model.onnx (single file)")

    weights = model.classifier[1].weight.detach().numpy()  # (3, 1280)
    with open("cam_weights.json", "w") as f:
        json.dump({"classes": CLASSES, "weights": weights.tolist()}, f)
    print("exported cam_weights.json", weights.shape)

    # --- verify: ONNX vs PyTorch on image-like inputs (pixels in [0,1],
    # normalised exactly like the app does). Raw out-of-range noise is not
    # used: float32 op-order differences amplify there without meaning
    # anything for real scans. ---
    import onnxruntime as ort

    sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    for trial in range(3):
        px = rng.random((1, 3, 224, 224), dtype=np.float32)
        x = ((px - mean) / std).astype(np.float32)
        with torch.no_grad():
            ref = torch.softmax(model(torch.tensor(x)), dim=1).numpy()[0]
        logits, acts = sess.run(None, {"input": x})
        got = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]
        diff = float(np.abs(ref - got).max())
        print(f"trial {trial}: prob max diff = {diff:.2e}")
        assert diff < 1e-4, "ONNX output differs from PyTorch!"
    print("verified: ONNX predictions match PyTorch")


if __name__ == "__main__":
    main()
