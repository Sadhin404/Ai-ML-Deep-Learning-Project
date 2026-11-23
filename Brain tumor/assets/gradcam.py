# Grad-CAM: a heatmap of the image regions that drove the model's decision.
# Hook-based, on the last dense block — the same implementation as the notebook.
import numpy as np
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def target_layer_of(model):
    return list(model.features.children())[-2]


def save_gradcam_examples(model, test_items, device, out_path):
    """One test image per class: Original | Heatmap | Overlay panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dataset import CLASSES, test_transform

    gradcam = GradCAM(model, target_layer_of(model))
    model.eval()

    picked = []
    for label in range(4):
        for path, y in test_items:
            if y == label:
                picked.append((path, y))
                break

    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for row, (path, label) in enumerate(picked):
        image = Image.open(path).convert('RGB')
        x = test_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred]) * 100

        cam = gradcam.generate(x, pred)
        cam_big = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0
        heatmap = plt.cm.jet(cam_big)[:, :, :3]
        original = np.array(image.resize((224, 224))) / 255.0
        overlay = 0.55 * heatmap + 0.45 * original

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f'Original: {CLASSES[label]}')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(cam_big, cmap='jet')
        axes[row, 1].set_title('Grad-CAM Heatmap')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f'Predicted: {CLASSES[pred]} ({conf:.1f}%)')
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Grad-CAM image saved:', out_path)


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dataset import CLASSES, test_transform
    from model import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--checkpoint', default='best_model.pth')
    ap.add_argument('--out', default='gradcam.png')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image = Image.open(args.image).convert('RGB')
    x = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred = int(probs.argmax())
    conf = float(probs[pred]) * 100

    cam = GradCAM(model, target_layer_of(model)).generate(x, pred)
    cam_big = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0
    heatmap = plt.cm.jet(cam_big)[:, :, :3]
    original = np.array(image.resize((224, 224))) / 255.0
    overlay = 0.55 * heatmap + 0.45 * original

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(cam_big, cmap='jet'); axes[1].set_title('Grad-CAM Heatmap'); axes[1].axis('off')
    axes[2].imshow(overlay); axes[2].set_title(f'Predicted: {CLASSES[pred]} ({conf:.1f}%)'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Predicted: {CLASSES[pred]} ({conf:.1f}%) — saved {args.out}')
