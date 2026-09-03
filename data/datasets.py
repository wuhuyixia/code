from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms


def extract_targets(dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        base_targets = extract_targets(dataset.dataset)
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        return base_targets[indices].long().clone()
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.long().clone()
        return torch.tensor(targets, dtype=torch.long)
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].long().clone()
    raise TypeError("Dataset must expose .targets or be a TensorDataset.")


def stratified_split(dataset, val_ratio: float, seed: int):
    """Deterministic class-stratified train/validation split."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")
    targets = extract_targets(dataset)
    generator = torch.Generator().manual_seed(int(seed))
    train_parts, val_parts = [], []
    for class_id in torch.unique(targets, sorted=True):
        indices = torch.where(targets == class_id)[0]
        if indices.numel() < 2:
            raise ValueError("Each class needs at least two samples for train/validation splitting.")
        order = torch.randperm(indices.numel(), generator=generator)
        indices = indices[order]
        n_val = max(1, int(round(indices.numel() * val_ratio)))
        n_val = min(n_val, indices.numel() - 1)
        val_parts.append(indices[:n_val])
        train_parts.append(indices[n_val:])
    train_idx = torch.cat(train_parts)
    val_idx = torch.cat(val_parts)
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def _build_transform(cfg: dict):
    """Build a transform entirely from explicit configuration.

    No dataset-specific normalization constants are hidden in this framework.
    """
    ops = []
    image_size = cfg.get("image_size")
    if image_size is not None:
        if isinstance(image_size, int):
            size = (image_size, image_size)
        else:
            size = tuple(int(v) for v in image_size)
        ops.append(transforms.Resize(size))
    if bool(cfg.get("to_tensor", True)):
        ops.append(transforms.ToTensor())
    mean = cfg.get("normalize_mean")
    std = cfg.get("normalize_std")
    if (mean is None) != (std is None):
        raise ValueError("normalize_mean and normalize_std must be provided together.")
    if mean is not None:
        ops.append(transforms.Normalize(tuple(mean), tuple(std)))
    return transforms.Compose(ops)


class GTSRBDataset(Dataset):
    """Read a local GTSRB Train.csv/Test.csv layout with optional ROI cropping."""

    def __init__(self, root: str | Path, split: str, transform=None, crop_roi: bool = True):
        self.root = Path(root)
        split = split.lower()
        if split not in {"train", "test"}:
            raise ValueError("GTSRB split must be 'train' or 'test'.")
        self.transform = transform
        self.crop_roi = bool(crop_roi)
        annotation = self.root / f"{split.capitalize()}.csv"
        if not annotation.is_file():
            raise FileNotFoundError(f"GTSRB annotation file not found: {annotation}")

        samples = []
        with annotation.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not {"Path", "ClassId"}.issubset(fields):
                raise ValueError(f"{annotation} must contain Path and ClassId columns.")
            roi_fields = ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
            has_roi = all(field in fields for field in roi_fields)
            for row in reader:
                roi = tuple(int(row[field]) for field in roi_fields) if has_roi else None
                rel_path = Path(str(row["Path"]).replace("\\", "/"))
                samples.append((self.root / rel_path, int(row["ClassId"]), roi))
        if not samples:
            raise ValueError(f"GTSRB annotation file is empty: {annotation}")
        self.samples = samples
        self.targets = torch.tensor([label for _, label, _ in samples], dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target, roi = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.crop_roi and roi is not None:
                x1, y1, x2, y2 = roi
                width, height = image.size
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2 + 1, width))
                y2 = max(y1 + 1, min(y2 + 1, height))
                image = image.crop((x1, y1, x2, y2))
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def load_dataset(cfg: dict, seed: int):
    """Load train/validation/test datasets from explicit configuration."""
    name = cfg.get("dataset") or cfg.get("name")
    if name is None:
        raise ValueError("Missing required data configuration: dataset")
    name = str(name).lower()
    root = Path(cfg.get("root", "data_cache"))
    root.mkdir(parents=True, exist_ok=True)
    download = bool(cfg.get("download", False))
    transform = _build_transform(cfg)

    if name == "mnist":
        train_full = datasets.MNIST(root=root, train=True, transform=transform, download=download)
        test = datasets.MNIST(root=root, train=False, transform=transform, download=download)
    elif name in {"fashion_mnist", "fashionmnist"}:
        train_full = datasets.FashionMNIST(root=root, train=True, transform=transform, download=download)
        test = datasets.FashionMNIST(root=root, train=False, transform=transform, download=download)
    elif name == "cifar10":
        train_full = datasets.CIFAR10(root=root, train=True, transform=transform, download=download)
        test = datasets.CIFAR10(root=root, train=False, transform=transform, download=download)
    elif name == "gtsrb":
        train_full = GTSRBDataset(root, "train", transform=transform, crop_roi=cfg.get("crop_roi", True))
        test = GTSRBDataset(root, "test", transform=transform, crop_roi=cfg.get("crop_roi", True))
    else:
        raise ValueError(f"Unknown dataset: {name}")

    val_ratio = cfg.get("val_fraction", cfg.get("val_ratio"))
    if val_ratio is None:
        raise ValueError("Missing required data configuration: val_fraction")
    train, val = stratified_split(train_full, float(val_ratio), seed)
    return train, val, test
