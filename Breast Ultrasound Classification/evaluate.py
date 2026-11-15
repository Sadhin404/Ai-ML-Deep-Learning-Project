import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from dataset import BUSIDataset, CLASSES
from model import build_model
from train import get_transforms, save_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", default="outputs/best_model.pth")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location=device)
    classes = ckpt.get("classes", CLASSES); size = ckpt.get("img_size", 224)
    split = json.loads((Path(a.output_dir) / "split_indices.json").read_text())
    ds = BUSIDataset(a.data_dir, get_transforms(size, train=False))
    assert [str(p) for p, _ in ds.samples] == split["sample_paths"], "Dataset order differs from saved split!"
    loader = DataLoader(Subset(ds, split["test"]), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=(device.type == "cuda"))
    model = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    metrics = save_evaluation(model, loader, device, out, classes)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
