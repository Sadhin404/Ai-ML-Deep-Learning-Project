# Predict one brain MRI image with the trained checkpoint.
# Usage: python predict.py --image your_image.jpg --checkpoint best_model.pth
import argparse
import torch
from PIL import Image

from dataset import CLASSES, test_transform
from model import build_model

DISPLAY = {'glioma': 'Glioma', 'meningioma': 'Meningioma', 'notumor': 'No Tumor', 'pituitary': 'Pituitary'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--checkpoint', default='best_model.pth')
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

    print(f'Prediction : {DISPLAY[CLASSES[pred]]} ({float(probs[pred]) * 100:.1f}% confidence)')
    for i, c in enumerate(CLASSES):
        print(f'  {DISPLAY[c]:12s} {float(probs[i]) * 100:5.1f}%')


if __name__ == '__main__':
    main()
