# Dataset loading for Brain Tumor Classification (DenseNet-201).
# Same code as the BrainTumor.ipynb notebook, organised as a module.
import os, random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def find_dataset(base):
    """Find the folder that contains Training/ and Testing/ (any depth)."""
    for root, dirs, files in os.walk(base):
        if 'Training' in dirs and 'Testing' in dirs:
            return root
    return None


def make_split(data_root, seed=42):
    """Training folder -> train/val (88/12, stratified, seeded).
    Testing folder -> the held-out test set (evaluated once, at the end)."""
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_items, val_items, test_items = [], [], []
    for label, c in enumerate(CLASSES):
        folder = os.path.join(data_root, 'Training', c)
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]
        rng.shuffle(files)
        n_val = int(round(len(files) * 0.12))
        val_items += [(f, label) for f in files[:n_val]]
        train_items += [(f, label) for f in files[n_val:]]

        folder = os.path.join(data_root, 'Testing', c)
        test_items += [(os.path.join(folder, f), label) for f in sorted(os.listdir(folder))]
    return train_items, val_items, test_items


# Image preprocessing + augmentation (augmentation only for training)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class BrainDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert('RGB')
        return self.transform(image), label


def make_loaders(train_items, val_items, test_items):
    train_loader = DataLoader(BrainDataset(train_items, train_transform), batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(BrainDataset(val_items, test_transform), batch_size=64, num_workers=2)
    test_loader = DataLoader(BrainDataset(test_items, test_transform), batch_size=64, num_workers=2)
    return train_loader, val_loader, test_loader
