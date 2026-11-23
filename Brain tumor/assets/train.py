# Full training run — the BrainTumor.ipynb notebook as a script.
# Trains DenseNet-201 for 20 epochs, keeps the best checkpoint by validation
# macro F1, evaluates the held-out test set once, and saves all outputs
# (best_model.pth, results.txt, confusion_matrix.png, training_curve.png,
# gradcam_examples.png) into the Result folder.
import argparse, os, shutil, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import CLASSES, find_dataset, make_split, make_loaders
from model import build_model
from evaluate import evaluate, report
from gradcam import save_gradcam_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data', help='folder containing the extracted dataset')
    ap.add_argument('--out', default='Result', help='where to save the outputs')
    ap.add_argument('--epochs', type=int, default=20)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    data_root = find_dataset(args.data)
    print('Dataset path:', data_root)
    train_items, val_items, test_items = make_split(data_root)
    print('Train:', len(train_items), '| Validation:', len(val_items), '| Test:', len(test_items))
    train_loader, val_loader, test_loader = make_loaders(train_items, val_items, test_items)

    model = build_model().to(device)

    epochs = args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1 = 0
    train_loss_history, val_acc_history = [], []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_loss, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total += len(y)
        scheduler.step()

        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        train_loss_history.append(total_loss / total)
        val_acc_history.append(val_acc)
        print(f'Epoch {epoch+1}/{epochs}  loss={total_loss/total:.4f}  val_acc={val_acc*100:.2f}%  val_f1={val_f1*100:.2f}%  ({time.time()-start:.0f}s)')

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            print('   -> best model saved')

    print('Training done. Best validation F1:', round(best_f1 * 100, 2), '%')

    # Final test — best model, evaluated once
    model.load_state_dict(torch.load('best_model.pth'))
    test_acc, test_f1, cm = evaluate(model, test_loader, device)
    print('TEST ACCURACY :', round(test_acc * 100, 2), '%')
    print('TEST MACRO F1 :', round(test_f1 * 100, 2), '%')
    print(cm)

    os.makedirs(args.out, exist_ok=True)

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - DenseNet-201')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i][j], ha='center', va='center',
                    color='white' if cm[i][j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Training curve figure
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(train_loss_history, color='red', label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss', color='red')
    ax2 = ax1.twinx()
    ax2.plot([a * 100 for a in val_acc_history], color='blue', label='Val Accuracy %')
    ax2.set_ylabel('Val Accuracy %', color='blue')
    plt.title('Training Curve - DenseNet-201')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'training_curve.png'), dpi=150)
    plt.close()

    # Grad-CAM panel (one test image per class)
    save_gradcam_examples(model, test_items, device, os.path.join(args.out, 'gradcam_examples.png'))

    # Results file + checkpoint
    with open(os.path.join(args.out, 'results.txt'), 'w') as f:
        f.write(report(cm))
    shutil.copy('best_model.pth', os.path.join(args.out, 'best_model.pth'))
    print('Saved to', args.out)


if __name__ == '__main__':
    main()
