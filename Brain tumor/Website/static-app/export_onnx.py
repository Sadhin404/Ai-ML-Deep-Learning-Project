#!/usr/bin/env python
"""Export Shadhin's trained DenseNet-201 (brain tumor) to ONNX for the web app.
Outputs: model_densenet201.onnx (fp32), model_densenet201_fp16.onnx, cam_weights.json
Verifies: PyTorch vs ONNX parity on local test images + analytic CAM == Grad-CAM check."""
import os, json, random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

ROOT = os.path.dirname(os.path.abspath(__file__))
PTH = os.path.join(ROOT, 'assets', 'best_model.pth')
DATA = os.path.expanduser('~/workspace/brain_tumor_project/data/raw')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
device = torch.device('cpu')

# ---------- find dataset ----------
def find_dataset(base):
    for root, dirs, files in os.walk(base):
        if 'Training' in dirs and 'Testing' in dirs:
            return root
    return None

data_root = find_dataset(DATA)
print('dataset:', data_root)

# ---------- build model exactly like the notebook ----------
model = models.densenet201(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier.in_features, 4))
sd = torch.load(PTH, map_location='cpu', weights_only=True)
model.load_state_dict(sd)
model.eval()
print('state dict loaded OK; classifier weight shape:', tuple(sd['classifier.1.weight'].shape))

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- sample test images: 50 per class from local Testing ----------
rng = random.Random(0)
test_items = []
for label, c in enumerate(CLASSES):
    folder = os.path.join(data_root, 'Testing', c)
    files = sorted(os.listdir(folder))
    rng.shuffle(files)
    test_items += [(os.path.join(folder, f), label) for f in files[:50]]
print('parity sample:', len(test_items), 'images')

def load_batch(items):
    xs, ys = [], []
    for p, y in items:
        xs.append(test_transform(Image.open(p).convert('RGB')))
        ys.append(y)
    return torch.stack(xs), np.array(ys)

X, Y = load_batch(test_items)

# ---------- PyTorch predictions ----------
with torch.no_grad():
    logits_pt = model(X).numpy()
pred_pt = logits_pt.argmax(1)
acc_pt = (pred_pt == Y).mean()
print(f'PyTorch accuracy on parity sample: {acc_pt*100:.1f}%')

# ---------- export wrapper: logits + post-ReLU feature maps ----------
class ExportWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        f = self.m.features(x)
        f = F.relu(f, inplace=True)
        pooled = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1)
        logits = self.m.classifier(pooled)
        return logits, f

wrapper = ExportWrapper(model).eval()
onnx_fp32 = os.path.join(ROOT, 'model_densenet201.onnx')
dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(wrapper, dummy, onnx_fp32, input_names=['input'],
                  output_names=['logits', 'acts'], opset_version=17, dynamo=False)
print('exported fp32:', onnx_fp32, os.path.getsize(onnx_fp32))

# ---------- fp16 conversion (io stays fp32, like the breast models) ----------
import onnx
from onnxconverter_common import float16
m = onnx.load(onnx_fp32)
m16 = float16.convert_float_to_float16(m, keep_io_types=True)
onnx_fp16 = os.path.join(ROOT, 'model_densenet201_fp16.onnx')
onnx.save(m16, onnx_fp16)
print('exported fp16:', onnx_fp16, os.path.getsize(onnx_fp16))

# ---------- ONNX parity ----------
import onnxruntime as ort
for name, path in [('fp32', onnx_fp32), ('fp16', onnx_fp16)]:
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    logits_on, acts_on = sess.run(['logits', 'acts'], {'input': X.numpy()})
    pred_on = logits_on.argmax(1)
    agree = (pred_on == pred_pt).mean()
    def softmax(v):
        e = np.exp(v - v.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    pdiff = np.abs(softmax(logits_on) - softmax(logits_pt)).max()
    print(f'ONNX {name}: argmax agreement with PyTorch = {agree*100:.1f}%, max prob diff = {pdiff:.4f}, acts shape = {acts_on.shape}')
    if name == 'fp16':
        acts_fp16 = acts_on

# ---------- CAM weights from the classifier ----------
W = sd['classifier.1.weight'].numpy()  # [4, 1920]
cam = {'classes': CLASSES, 'models': {'densenet201': [W[i].tolist() for i in range(4)]}}
with open(os.path.join(ROOT, 'cam_weights.json'), 'w') as f:
    json.dump(cam, f)
print('cam_weights.json written')

# ---------- analytic CAM == Grad-CAM numeric check (one image) ----------
path0, y0 = test_items[0]
x1 = test_transform(Image.open(path0).convert('RGB')).unsqueeze(0)
acts_store, grads_store = {}, {}
def fh(mod, inp, out): acts_store['a'] = out
def bh(mod, gin, gout): grads_store['g'] = gout[0]
target = list(model.features.children())[-2]
h1 = target.register_forward_hook(fh)
h2 = target.register_full_backward_hook(bh)
model.zero_grad()
out = model(x1)
pred0 = int(out.argmax(1))
out[0, pred0].backward()
h1.remove(); h2.remove()
g = grads_store['g']
w_gc = g.mean(dim=(2, 3)).squeeze(0).detach().numpy()          # Grad-CAM channel weights (denseblock4 out)
# analytic: classifier weights are w.r.t. norm5+ReLU output, Grad-CAM here is on denseblock4;
# compare instead against Grad-CAM computed on the export features (norm5 output):
acts_store2, grads_store2 = {}, {}
def fh2(mod, inp, out): acts_store2['a'] = out
def bh2(mod, gin, gout): grads_store2['g'] = gout[0]
target2 = list(model.features.children())[-1]
h1 = target2.register_forward_hook(fh2)
h2 = target2.register_full_backward_hook(bh2)
model.zero_grad()
out = model(x1)
out[0, pred0].backward()
h1.remove(); h2.remove()
w_gc2 = grads_store2['g'].mean(dim=(2, 3)).squeeze(0).detach().numpy()
w_an = W[pred0]
corr = np.corrcoef(w_gc2, w_an)[0, 1]
ratio = np.median(np.abs(w_gc2)) / np.median(np.abs(w_an))
print(f'Grad-CAM vs analytic weights: correlation = {corr:.4f} (expect ~1.0), median|w| ratio = {ratio:.3f} (expect ~1/49={1/49:.3f} up to ReLU effects)')

# ---------- gate calibration stats on local MRI images (JS gate algorithm) ----------
def gate_stats(path):
    im = Image.open(path).convert('RGB')
    scale = min(1.0, 256 / max(im.width, im.height))
    w = max(1, round(im.width * scale)); h = max(1, round(im.height * scale))
    im = im.resize((w, h))
    d = np.asarray(im).astype(np.float32)
    r, g, b = d[:, :, 0], d[:, :, 1], d[:, :, 2]
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    N = w * h
    color = (sat > 32).mean(); mean_sat = sat.mean(); bright = (lum > 200).mean()
    diffs = np.abs(np.diff(lum, axis=1))
    speckle = diffs.mean()
    flat = (np.diff(lum, axis=1) == 0).mean()
    return color, mean_sat, bright, speckle, flat, lum.std()

stats = []
for p, y in test_items[:40]:
    stats.append(gate_stats(p))
stats = np.array(stats)
names = ['colorFrac', 'meanSat', 'brightFrac', 'speckle', 'flatFrac', 'lumStd']
for i, n in enumerate(names):
    print(f'gate {n}: MRI min={stats[:,i].min():.3f} max={stats[:,i].max():.3f} mean={stats[:,i].mean():.3f}')
print('DONE')
