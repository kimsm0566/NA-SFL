"""CIFAR array loader and partition metadata used by NA-SFL."""
from __future__ import annotations
import hashlib
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def partition_metadata(
    labels: Sequence[int] | np.ndarray,
    partitions: Sequence[Sequence[int] | np.ndarray],
    partition: str,
    alpha: float,
    seed: int,
) -> dict:
    labels_array = np.asarray(labels, dtype=np.int64)
    clients = []
    for client_id, indices in enumerate(partitions):
        ids = np.sort(np.asarray(indices, dtype="<i8"))
        clients.append(
            {
                "client_id": client_id,
                "count": int(len(ids)),
                "sample_id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "class_counts": np.bincount(
                    labels_array[ids.astype(np.int64)], minlength=max(10, int(labels_array.max()) + 1)
                )
                .astype(int)
                .tolist(),
            }
        )
    return {
        "partition": partition,
        "seed": int(seed),
        "alpha": float(alpha) if partition == "dirichlet" else None,
        "clients": clients,
    }


class CIFARArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        sample_ids: np.ndarray,
        transform,
    ) -> None:
        self.images = images
        self.labels = labels
        self.sample_ids = sample_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = np.asarray(self.images[index])
        if image.shape == (3, 32, 32):
            image = image.transpose(1, 2, 0)
        if image.shape != (32, 32, 3):
            raise ValueError(f"expected a CIFAR image, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image * 255 if image.max() <= 1 else image, 0, 255).astype(
                np.uint8
            )
        return (
            self.transform(Image.fromarray(image, mode="RGB")),
            int(self.labels[index]),
            int(self.sample_ids[index]),
        )


def build_loader(
    images: Sequence | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    sample_ids: Sequence[int] | np.ndarray,
    batch_size: int,
    train: bool,
    num_workers: int,
    generator: torch.Generator,
) -> DataLoader:
    images_array = np.asarray(images)
    labels_array = np.asarray(labels, dtype=np.int64)
    ids_array = np.asarray(sample_ids, dtype=np.int64)
    if not len(labels_array) or len(images_array) != len(labels_array):
        raise ValueError("images and labels must be matching and nonempty")
    if len(ids_array) != len(labels_array):
        raise ValueError("sample_ids must match labels")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")

    operations = []
    if train:
        operations.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    dataset = CIFARArrayDataset(
        images_array,
        labels_array,
        ids_array,
        transforms.Compose(operations),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )
