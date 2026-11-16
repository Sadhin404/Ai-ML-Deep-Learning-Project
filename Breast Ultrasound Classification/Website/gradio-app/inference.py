"""Model loading, prediction and Grad-CAM for the web app.

Uses the exact same preprocessing and Grad-CAM method as the thesis
project (Resize 224x224, ImageNet normalisation, target layer
model.features[-1]).
"""
import io
import threading

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

CLASSES = ["normal", "benign", "malignant"]
DISPLAY = {"normal": "Normal", "benign": "Benign", "malignant": "Malignant"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_model = None
_lock = threading.Lock()


def _build_model(num_classes=3):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path):
    global _model
    if _model is not None:
        return _model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _build_model(len(ckpt.get("classes", CLASSES)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _model = model
    return _model


class _GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._fwd)
        self.h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()[0]
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, probs.detach().cpu().numpy()


def _focus_box(cam):
    """Bounding box (relative 0-1 coords) around the region that influenced
    the prediction most, derived from the Grad-CAM heatmap."""
    h, w = cam.shape
    mask = cam >= 0.55
    if mask.sum() < 0.01 * h * w:
        mask = cam >= 0.40
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw > 0.85 * w or bh > 0.85 * h:
        # Attention too diffuse — centre a box on the strongest point.
        py, px = np.unravel_index(int(cam.argmax()), cam.shape)
        bw, bh = int(0.46 * w), int(0.46 * h)
        x0, y0 = px - bw // 2, py - bh // 2
        x1, y1 = x0 + bw, y0 + bh
    pad = 8
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad); y1 = min(h - 1, y1 + pad)
    return {"x": x0 / w, "y": y0 / h, "w": (x1 - x0 + 1) / w, "h": (y1 - y0 + 1) / h}


def _jet_colormap(arr01):
    """Small jet colormap (matches the thesis panels, which used matplotlib jet)."""
    x = np.clip(arr01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def analyse(image_pil):
    """Run prediction + Grad-CAM. Returns a dict with results and PIL images."""
    model = load_model(_checkpoint_path)
    image = image_pil.convert("RGB")
    x = _transform(image).unsqueeze(0)
    with _lock:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        cam_gen = _GradCAM(model, model.features[-1])
        try:
            cam, probs = cam_gen(x, pred)
        finally:
            cam_gen.close()

    disp = image.resize((IMG_SIZE, IMG_SIZE))
    orig_np = np.asarray(disp)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    cam_big = np.asarray(cam_img) / 255.0
    heat_rgb = _jet_colormap(cam_big)
    overlay = (0.55 * orig_np + 0.45 * heat_rgb).astype(np.uint8)

    return {
        "image_info": {
            "width": image_pil.width,
            "height": image_pil.height,
            "mode": image_pil.mode,
            "format": image_pil.format or "—",
        },
        "bbox": _focus_box(cam_big),
        "predicted_class": CLASSES[pred],
        "predicted_display": DISPLAY[CLASSES[pred]],
        "confidence": float(probs[pred]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)},
        "images": {
            "original": disp.copy(),
            "heatmap": Image.fromarray(heat_rgb),
            "overlay": Image.fromarray(overlay),
        },
    }


_checkpoint_path = "model/best_model.pth"


def init(checkpoint_path):
    global _checkpoint_path
    _checkpoint_path = checkpoint_path
    load_model(checkpoint_path)


def pil_to_png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
