import argparse, torch
from PIL import Image
from torchvision import transforms
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="outputs/best_model.pth")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device)
    classes = ckpt.get("classes", ["normal", "benign", "malignant"])
    size = ckpt.get("img_size", 224)
    model = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = Image.open(a.image).convert("RGB")
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0).to(device)), dim=1)[0]
    for i in probs.argsort(descending=True).tolist():
        print(f"{classes[i]}: {probs[i].item()*100:.2f}%")


if __name__ == "__main__":
    main()
