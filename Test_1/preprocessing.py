
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class PreprocessingConfig:
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 0
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


class DigitsImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, image_size: int = 64) -> None:
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.images[idx] / 16.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)   # [1, 8, 8]
        image = image.repeat(3, 1, 1)                                   # [3, 8, 8]
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        label = int(self.labels[idx])
        return image, label


def build_digits_dataloaders(cfg: PreprocessingConfig):
    data = load_digits()
    images = data.images
    labels = data.target
    class_names: List[str] = [str(x) for x in data.target_names.tolist()]

    x_train, x_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        test_size=cfg.val_size + cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    relative_test = cfg.test_size / (cfg.val_size + cfg.test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test,
        random_state=cfg.random_state,
        stratify=y_temp,
    )

    train_ds = DigitsImageDataset(x_train, y_train, image_size=cfg.image_size)
    val_ds = DigitsImageDataset(x_val, y_val, image_size=cfg.image_size)
    test_ds = DigitsImageDataset(x_test, y_test, image_size=cfg.image_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader, test_loader, class_names
