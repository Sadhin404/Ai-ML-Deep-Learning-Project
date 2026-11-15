from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["normal", "benign", "malignant"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def scan_dataset(root):
    root = Path(root)
    samples = []
    stats = {}
    class_to_idx = {name: i for i, name in enumerate(CLASSES)}
    for cls in CLASSES:
        folder = root / cls
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {folder}. "
                "Set --data_dir to the directory containing normal, benign, malignant."
            )
        valid, masks, other = 0, 0, 0
        for p in sorted(folder.iterdir()):
            if not p.is_file():
                continue
            if "_mask" in p.stem.lower():
                masks += 1
            elif p.suffix.lower() in IMAGE_EXTS:
                samples.append((p, class_to_idx[cls]))
                valid += 1
            else:
                other += 1
        stats[cls] = {"valid_images": valid, "mask_files_excluded": masks, "other_files": other}
    return samples, stats


class BUSIDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples, self.stats = scan_dataset(root)
        self.class_to_idx = {name: i for i, name in enumerate(CLASSES)}
        if not self.samples:
            raise RuntimeError(f"No usable images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, str(path)
