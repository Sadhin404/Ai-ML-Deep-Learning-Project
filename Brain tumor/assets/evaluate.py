# Test-set evaluation: accuracy, per-class precision/recall/F1, confusion matrix.
import numpy as np
import torch

from dataset import CLASSES


def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((4, 4), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                cm[t][p] += 1
    acc = np.trace(cm) / cm.sum()
    f1_list = []
    for i in range(4):
        tp = cm[i][i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = tp / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)
    return acc, float(np.mean(f1_list)), cm


def report(cm):
    """The results.txt content for a confusion matrix (same format as the notebook)."""
    lines = []
    lines.append('Brain Tumor Classification - DenseNet-201')
    lines.append('Dataset: Brain Tumor MRI Dataset (Kaggle, original split)')
    lines.append('')
    acc = np.trace(cm) / cm.sum()
    f1s = []
    rows = []
    for i, c in enumerate(CLASSES):
        tp = cm[i][i]
        precision = tp / cm[:, i].sum() * 100
        recall = tp / cm[i, :].sum() * 100
        f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)
        rows.append(f'{c:12s} {precision:.2f} / {recall:.2f} / {f1:.2f} / {cm[i, :].sum()}')
    lines.append(f'Test Accuracy : {acc * 100:.2f}%')
    lines.append(f'Test Macro F1 : {np.mean(f1s):.2f}%')
    lines.append('')
    lines.append('Class wise (Precision / Recall / F1 / Total):')
    lines += rows
    lines.append('')
    lines.append('Confusion Matrix:')
    lines.append(str(cm))
    lines.append('')
    lines.append(f'Errors: {int(cm.sum() - np.trace(cm))} / {int(cm.sum())}')
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse, os
    from dataset import find_dataset, make_split, make_loaders
    from model import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data', help='folder containing the extracted dataset')
    ap.add_argument('--checkpoint', default='best_model.pth')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = find_dataset(args.data)
    _, _, test_items = make_split(data_root)
    _, _, test_loader = make_loaders([], [], test_items)

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    acc, f1, cm = evaluate(model, test_loader, device)
    print(report(cm))
