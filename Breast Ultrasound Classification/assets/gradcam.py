import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dataset import BUSIDataset, CLASSES
from model import build_model
from train import get_transforms

TARGET_LAYER_NAME = "features[-1] (final Conv2dNormActivation, 1280 ch)"


class GradCAM:
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
        self.h1.remove(); self.h2.remove()

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


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    classes = ckpt.get("classes", CLASSES)
    size = ckpt.get("img_size", 224)
    model = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model, classes, size


def save_panel(image_pil, cam, true_name, pred_name, confidence, probs, classes, output_path):
    size = image_pil.size[0]
    heat = Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.BILINEAR)
    heat_np = np.asarray(heat)
    orig_np = np.asarray(image_pil.convert("RGB").resize((size, size)))
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heat_np / 255.0)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * orig_np + 0.45 * heat_rgb).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(orig_np); axes[0].set_title("Original ultrasound"); axes[0].axis("off")
    axes[1].imshow(heat_np, cmap="jet"); axes[1].set_title("Grad-CAM heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Grad-CAM overlay"); axes[2].axis("off")
    prob_txt = "  |  ".join(f"{c}: {p*100:.1f}%" for c, p in zip(classes, probs))
    fig.suptitle(f"True: {true_name}   Predicted: {pred_name}   Confidence: {confidence*100:.2f}%\n{prob_txt}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200); plt.close(fig)


def run_single(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, size = load_model(args.checkpoint, device)
    cam_gen = GradCAM(model, model.features[-1])
    image = Image.open(args.image).convert("RGB")
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    x = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = int(torch.softmax(model(x), dim=1).argmax(1).item())
    cam, probs = cam_gen(x, pred)
    cam_gen.close()
    save_panel(image.resize((size, size)), cam, "(unknown)", classes[pred], float(probs[pred]),
               probs, classes, args.output)
    print(f"Prediction: {classes[pred]} ({probs[pred]*100:.2f}%) -> {args.output}")


def run_batch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, size = load_model(args.checkpoint, device)
    split = json.loads(Path(args.split_file).read_text())
    ds = BUSIDataset(args.data_dir, get_transforms(size, train=False))
    assert [str(p) for p, _ in ds.samples] == split["sample_paths"], "Dataset order differs from saved split!"
    test_idx = split["test"]
    cam_gen = GradCAM(model, model.features[-1])
    out_root = Path(args.output_dir)
    rows = []
    per_class_done = {c: 0 for c in classes}
    incorrect_done = 0
    for idx in test_idx:
        path, true_label = ds.samples[idx]
        image = Image.open(path).convert("RGB")
        x = get_transforms(size, train=False)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs0 = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        pred = int(probs0.argmax()); conf = float(probs0[pred])
        correct = int(pred == true_label)
        want_class = per_class_done[classes[true_label]] < args.per_class
        want_incorrect = (not correct) and incorrect_done < args.per_class
        if not (want_class or want_incorrect or correct):
            continue
        cam, probs = cam_gen(x, pred)
        stem = Path(path).stem.replace(" ", "_")
        fname = f"{classes[true_label]}__{stem}__pred-{classes[pred]}__conf-{conf:.3f}.png"
        disp = image.resize((size, size))
        saved_to = []
        if want_class:
            pth = out_root / classes[true_label].capitalize() / fname
            save_panel(disp, cam, classes[true_label], classes[pred], conf, probs, classes, pth)
            saved_to.append(str(pth)); per_class_done[classes[true_label]] += 1
        folder = "Correct_Predictions" if correct else "Incorrect_Predictions"
        if correct or want_incorrect:
            pth = out_root / folder / fname
            save_panel(disp, cam, classes[true_label], classes[pred], conf, probs, classes, pth)
            saved_to.append(str(pth))
            if not correct: incorrect_done += 1
        rows.append({"image": str(path), "true_class": classes[true_label],
                     "predicted_class": classes[pred], "confidence": f"{conf:.6f}",
                     "correct": correct, "files": ";".join(saved_to)} |
                    {f"prob_{c}": f"{float(v):.6f}" for c, v in zip(classes, probs)})
        if all(v >= args.per_class for v in per_class_done.values()) and incorrect_done >= args.per_class:
            break
    cam_gen.close()
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "gradcam_summary.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"Grad-CAM batch done: {len(rows)} visualisations, per-class {per_class_done}, incorrect {incorrect_done}")
    print("Summary:", out_root / "gradcam_summary.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image"); p.add_argument("--checkpoint", default="outputs/best_model.pth")
    p.add_argument("--output", default="gradcam_result.png")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--data_dir"); p.add_argument("--split_file", default="outputs/split_indices.json")
    p.add_argument("--per_class", type=int, default=5)
    p.add_argument("--output_dir", default="outputs/gradcam")
    a = p.parse_args()
    if a.batch:
        if not a.data_dir: p.error("--data_dir is required with --batch")
        run_batch(a)
    elif a.image:
        run_single(a)
    else:
        p.error("Give --image for a single image or --batch for the test set")


if __name__ == "__main__":
    main()
