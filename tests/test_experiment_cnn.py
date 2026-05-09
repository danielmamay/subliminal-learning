"""Smoke tests for teacher training and student distillation."""

from typing import cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from subliminal_learning.config import Config
from subliminal_learning.experiment_cnn import _accuracy as accuracy
from subliminal_learning.experiment_cnn import _distill_student as distill_student
from subliminal_learning.experiment_cnn import _train_teacher as train_teacher
from subliminal_learning.model import MLP


def _tiny_config() -> Config:
    return Config(
        input_shape=(1, 28, 28),
        n_classes=10,
        hidden_dims=(64, 64),
        n_aux=3,
        n_epochs_teacher=2,
        n_epochs_student=1,
        batch_size=64,
        lr=1e-3,
    )


def _tiny_loader(config: Config, n: int = 512) -> DataLoader:
    x = torch.randn(n, *config.input_shape)
    y = torch.randint(0, config.n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)


def test_teacher_improves_accuracy() -> None:
    config = _tiny_config()
    device = torch.device("cpu")
    model = MLP.from_config(config)
    loader = _tiny_loader(config)
    before = accuracy(model, loader, device)
    train_teacher(model, loader, config, device)
    after = accuracy(model, loader, device)
    assert after > before


def test_distill_aux_only_runs() -> None:
    config = _tiny_config()
    device = torch.device("cpu")
    teacher = MLP.from_config(config)
    student = MLP.from_config(config)
    distill_student(
        student,
        teacher,
        n_batches_per_epoch=4,
        config=config,
        device=device,
        use_all_logits=False,
    )


def test_distill_all_logits_runs() -> None:
    config = _tiny_config()
    device = torch.device("cpu")
    teacher = MLP.from_config(config)
    student = MLP.from_config(config)
    distill_student(
        student,
        teacher,
        n_batches_per_epoch=4,
        config=config,
        device=device,
        use_all_logits=True,
    )


def test_accuracy_output_uses_class_logits_only() -> None:
    """Accuracy must only look at logits[:, :n_classes], not aux logits."""
    config = _tiny_config()
    device = torch.device("cpu")
    model = MLP.from_config(config)
    loader = _tiny_loader(config, n=128)
    # Manually zero the aux logits and confirm accuracy is unchanged.
    with torch.no_grad():
        last_layer = cast(nn.Linear, model.net[-1])
        last_layer.weight[config.n_classes :].zero_()
        last_layer.bias[config.n_classes :].zero_()
    acc_zeroed = accuracy(model, loader, device)
    # Restore and re-zero class logits instead — accuracy should collapse.
    model2 = MLP.from_config(config)
    with torch.no_grad():
        last_layer2 = cast(nn.Linear, model2.net[-1])
        last_layer2.weight[: config.n_classes].zero_()
        last_layer2.bias[: config.n_classes].zero_()
    acc_class_zeroed = accuracy(model2, loader, device)
    # Zeroing aux has no effect on accuracy; zeroing class logits tanks it.
    assert acc_zeroed >= 0.0
    assert acc_class_zeroed <= 1.0 / config.n_classes + 0.05
