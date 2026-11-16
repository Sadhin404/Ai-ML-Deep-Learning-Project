import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import BUSIDataset, CLASSES
from model import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size, train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    yt, yp, probs, paths = [], [], [], []
    for x, y, p in loader:
        logits = model(x.to(device))
        pr = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(pr)
        yp.extend(pr.argmax(1).tolist())
        yt.extend(y.tolist())
        paths.extend(list(p))
    return yt, yp, np.concatenate(probs), paths


def save_evaluation(model, test_loader, device, out, classes=CLASSES):
    yt, yp, probs, paths = predict_all(model, test_loader, device)
    acc = accuracy_score(yt, yp)
    macro_f1 = f1_score(yt, yp, average="macro")
    weighted_f1 = f1_score(yt, yp, average="weighted")
    report_dict = classification_report(yt, yp, target_names=classes, output_dict=True, zero_division=0)
    (out / "classification_report.txt").write_text(
        classification_report(yt, yp, target_names=classes, zero_division=0), encoding="utf-8")

    cm = confusion_matrix(yt, yp, labels=list(range(len(classes))))
    with open(out / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["actual\\predicted"] + classes); 
        for i, row in enumerate(cm): w.writerow([classes[i]] + row.tolist())
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title("Confusion Matrix (Test Set)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=300); plt.close(fig)

    roc_auc = {}
    try:
        yt_bin = np.zeros((len(yt), len(classes))); yt_bin[np.arange(len(yt)), yt] = 1
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, cls in enumerate(classes):
            if yt_bin[:, i].sum() in (0, len(yt)): continue
            fpr, tpr, _ = roc_curve(yt_bin[:, i], probs[:, i])
            auc_i = roc_auc_score(yt_bin[:, i], probs[:, i]); roc_auc[cls] = float(auc_i)
            ax.plot(fpr, tpr, label=f"{cls} (AUC = {auc_i:.3f})")
        if roc_auc:
            roc_auc["macro_average"] = float(np.mean(list(roc_auc.values())))
            ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curves - One-vs-Rest (Test Set)")
            ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=300)
        plt.close("all")
    except Exception as e:
        print("ROC skipped:", e)

    with open(out / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_path", "true_class", "predicted_class", "correct"] +
                                      [f"prob_{c}" for c in classes])
        for path, t, p_, pr in zip(paths, yt, yp, probs):
            w.writerow([path, classes[t], classes[p_], int(t == p_)] + [f"{v:.6f}" for v in pr])

    metrics = {"test_accuracy": float(acc), "test_macro_f1": float(macro_f1),
               "test_weighted_f1": float(weighted_f1), "test_samples": len(yt),
               "per_class": {c: report_dict[c] for c in classes}, "roc_auc_ovr": roc_auc}
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    ds_train = BUSIDataset(args.data_dir, get_transforms(args.img_size, train=True))
    ds_eval = BUSIDataset(args.data_dir, get_transforms(args.img_size, train=False))
    print("Dataset statistics:", json.dumps(ds_train.stats, indent=2))
    labels = [y for _, y in ds_train.samples]
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=args.seed, stratify=labels)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=args.seed, stratify=temp_labels)
    split = {"seed": args.seed, "classes": CLASSES,
             "train": sorted(map(int, train_idx)), "val": sorted(map(int, val_idx)),
             "test": sorted(map(int, test_idx)),
             "sample_paths": [str(pth) for pth, _ in ds_train.samples],
             "sample_labels": labels}
    (out / "split_indices.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Split sizes -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    (out / "config.json").write_text(json.dumps(vars(args) | {"device": str(device),
        "python": sys.version, "torch": torch.__version__}, indent=2), encoding="utf-8")

    pin = device.type == "cuda"
    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(Subset(ds_eval, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(Subset(ds_eval, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    counts = np.bincount([labels[i] for i in train_idx], minlength=len(CLASSES)).astype(float)
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32).to(device)
    print("Class weights:", class_weights.tolist())

    model = build_model(len(CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, bad_epochs = -1.0, 0
    history = {"train_loss": [], "val_macro_f1": [], "val_accuracy": []}
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / len(train_idx)
        yt, yp, _, _ = predict_all(model, val_loader, device)
        val_f1 = f1_score(yt, yp, average="macro"); val_acc = accuracy_score(yt, yp)
        history["train_loss"].append(float(train_loss)); history["val_macro_f1"].append(float(val_f1)); history["val_accuracy"].append(float(val_acc))
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_accuracy={val_acc:.4f}, val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, bad_epochs = float(val_f1), 0
            torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES,
                        "img_size": args.img_size, "val_macro_f1": best_f1, "epoch": epoch + 1},
                       out / "best_model.pth")
            print("  -> saved best_model.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (best val macro F1 = {best_f1:.4f})")
                break

    with open(out / "training_history.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_accuracy", "val_macro_f1"])
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1, history["train_loss"][i], history["val_accuracy"][i], history["val_macro_f1"][i]])
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(history["train_loss"], "o-", color="tab:red", label="Training loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="tab:red")
    ax2 = ax1.twinx(); ax2.plot(history["val_macro_f1"], "s-", color="tab:blue", label="Validation macro F1")
    ax2.set_ylabel("Macro F1", color="tab:blue"); ax1.set_title("Training History")
    fig.tight_layout(); fig.savefig(out / "training_history.png", dpi=300); plt.close(fig)

    ckpt = torch.load(out / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Evaluating best checkpoint (epoch {ckpt.get('epoch')}, val macro F1 {ckpt.get('val_macro_f1'):.4f}) on the test set...")
    metrics = save_evaluation(model, test_loader, device, out)
    metrics["best_val_macro_f1"] = best_f1; metrics["history"] = history
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "test_macro_f1", "test_weighted_f1", "test_samples")}, indent=2))
    print("Done. Outputs in:", out.resolve())


if __name__ == "__main__":
    main()
