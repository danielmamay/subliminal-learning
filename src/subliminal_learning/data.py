"""Dataset loaders and noise input generation."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class LabeledLoader(Protocol):
    """Minimal interface for a loader that yields ``(inputs, labels)`` batches."""

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]: ...
    def __len__(self) -> int: ...


class TensorLoader:
    """On-device data loader backed by pre-loaded tensors.

    Keeps all data on the target device, eliminating per-batch CPU→device
    transfers and DataLoader multiprocessing overhead.
    """

    def __init__(self, x: Tensor, y: Tensor, batch_size: int, *, shuffle: bool) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(self.x.size(0) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        n = self.x.size(0)
        idx = (
            torch.randperm(n, device=self.x.device)
            if self.shuffle
            else torch.arange(n, device=self.x.device)
        )
        for start in range(0, n, self.batch_size):
            batch = idx[start : start + self.batch_size]
            yield self.x[batch], self.y[batch]


@dataclass(frozen=True)
class DatasetConfig:
    """Static metadata for a supported classification dataset."""

    input_shape: tuple[int, ...]  # (C, H, W)
    n_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    dataset_cls: type  # torchvision dataset class


#: Registry of supported datasets.
DATASETS: dict[str, DatasetConfig] = {
    "mnist": DatasetConfig(
        input_shape=(1, 28, 28),
        n_classes=10,
        mean=(0.1307,),
        std=(0.3081,),
        dataset_cls=datasets.MNIST,
    ),
    "fashion-mnist": DatasetConfig(
        input_shape=(1, 28, 28),
        n_classes=10,
        mean=(0.2860,),
        std=(0.3530,),
        dataset_cls=datasets.FashionMNIST,
    ),
    "cifar-10": DatasetConfig(
        input_shape=(3, 32, 32),
        n_classes=10,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        dataset_cls=datasets.CIFAR10,
    ),
}


def get_loaders(
    name: str,
    batch_size: int,
    device: torch.device,
    data_dir: Path = Path("./data"),
) -> tuple[TensorLoader, TensorLoader]:
    """Load a dataset into device memory and return on-device ``(train, test)`` loaders.

    Images are returned as ``[batch, C, H, W]`` tensors normalised to zero mean
    and unit variance.
    """
    cfg = DATASETS[name]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    train_ds = cfg.dataset_cls(data_dir, train=True, download=True, transform=transform)
    test_ds = cfg.dataset_cls(data_dir, train=False, download=True, transform=transform)

    def _collect(ds: Dataset[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        for x, y in loader:
            xs.append(x)
            ys.append(y)
        return torch.cat(xs).to(device), torch.cat(ys).to(device)

    x_train, y_train = _collect(train_ds)
    x_test, y_test = _collect(test_ds)
    return (
        TensorLoader(x_train, y_train, batch_size, shuffle=True),
        TensorLoader(x_test, y_test, batch_size, shuffle=False),
    )


def sample_noise(
    batch_size: int, input_shape: tuple[int, ...], device: torch.device
) -> Tensor:
    """Sample a batch of standard-normal noise with the given input shape."""
    return torch.randn(batch_size, *input_shape, device=device)
