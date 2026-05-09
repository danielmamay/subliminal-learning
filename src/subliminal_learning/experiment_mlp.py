"""MLP experiment: all N seeds trained simultaneously via batched matmul.

Uses :class:`~subliminal_learning.model.MultiMLP` to run all N seeds in a
single GPU kernel per layer with a standard :class:`torch.optim.Adam` optimizer.
"""

import copy
import dataclasses
import math
import statistics

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from .config import Config
from .data import LabeledLoader, sample_noise
from .model import MultiMLP


def _train_teachers(
    multi: MultiMLP,
    train_loader: LabeledLoader,
    config: Config,
    name: str = "teacher",
) -> MultiMLP:
    multi = copy.deepcopy(multi)
    opt = torch.optim.Adam(multi.parameters(), lr=config.lr)
    n = multi.n_models
    n_classes = config.n_classes

    for epoch in range(config.n_epochs_teacher):
        epoch_loss = epoch_acc = 0.0
        for x, y in train_loader:
            logits = multi(x)  # [n, batch, n_out]
            batch_size = y.size(0)
            class_logits = logits[:, :, :n_classes].reshape(n * batch_size, n_classes)
            labels = y.unsqueeze(0).expand(n, -1).reshape(n * batch_size)
            loss = F.cross_entropy(class_logits, labels)
            acc = (class_logits.detach().argmax(-1) == labels).float().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        n_batches = len(train_loader)
        logger.debug(
            f"{name} | epoch {epoch + 1}/{config.n_epochs_teacher} "
            f"| loss {epoch_loss / n_batches:.4f} | acc {epoch_acc / n_batches:.3f}"
        )

    return multi


def _distill_students(
    student_init: MultiMLP,
    teacher: MultiMLP,
    n_batches_per_epoch: int,
    config: Config,
    device: torch.device,
    *,
    use_all_logits: bool,
    name: str = "student",
) -> MultiMLP:
    student = copy.deepcopy(student_init)
    opt = torch.optim.Adam(student.parameters(), lr=config.lr)
    n = student.n_models
    n_classes = config.n_classes
    T = config.temperature

    for epoch in range(config.n_epochs_student):
        epoch_loss = 0.0
        for _ in range(n_batches_per_epoch):
            x = sample_noise(config.batch_size, config.input_shape, device)
            with torch.no_grad():
                t_logits = teacher(x)  # [n, batch, n_out]
            s_logits = student(x)  # [n, batch, n_out]

            t_slice = t_logits if use_all_logits else t_logits[:, :, n_classes:]
            s_slice = s_logits if use_all_logits else s_logits[:, :, n_classes:]

            n_out = s_slice.shape[-1]
            batch_size = x.shape[0]
            loss = F.kl_div(
                F.log_softmax(s_slice.reshape(n * batch_size, n_out) / T, dim=-1),
                F.softmax(t_slice.reshape(n * batch_size, n_out) / T, dim=-1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        logger.debug(
            f"{name} | epoch {epoch + 1}/{config.n_epochs_student} "
            f"| loss {epoch_loss / n_batches_per_epoch:.4f}"
        )

    return student


@torch.no_grad()
def _evaluate_n(
    multi: MultiMLP,
    loader: LabeledLoader,
    n_classes: int,
    device: torch.device,
) -> Tensor:
    """Evaluate N models simultaneously; returns ``[N]`` accuracy tensor."""
    n = multi.n_models
    correct = torch.zeros(n, device=device)
    total = 0
    for x, y in loader:
        logits = multi(x)  # [n, batch, n_out]
        preds = logits[:, :, :n_classes].argmax(-1)  # [n, batch]
        correct += (preds == y.unsqueeze(0)).float().sum(-1)
        total += y.size(0)
    return correct / total


def run_experiment_mlp(
    config: Config,
    device: torch.device,
    train_loader: LabeledLoader,
    test_loader: LabeledLoader,
    model_cls: type,
    *,
    use_wandb: bool = True,
    wandb_project: str = "subliminal-learning-vision",
) -> dict[str, list[float]]:
    """Run the full experiment with all ``config.n_seeds`` seeds in parallel.

    Drop-in replacement for :func:`~subliminal_learning.experiment.run_experiment`.
    All N models are trained simultaneously via
    :class:`~subliminal_learning.model.MultiMLP`.
    Only valid for MLP; the sequential runner handles other architectures.
    """
    import wandb

    if use_wandb:
        wandb.init(project=wandb_project, config=dataclasses.asdict(config))

    n = config.n_seeds
    n_batches = len(train_loader)

    logger.info(f"Initialising {n} reference model pairs...")
    ref_a = MultiMLP.from_seeds(list(range(n)), config).to(device)
    ref_b = MultiMLP.from_seeds([s + 1_000_000 for s in range(n)], config).to(device)

    logger.info("Training same-init teachers...")
    teacher_same = _train_teachers(
        ref_a, train_loader, config, name="teacher (same init)"
    )
    logger.info("Training diff-init teachers...")
    teacher_diff = _train_teachers(
        ref_b, train_loader, config, name="teacher (diff init)"
    )

    logger.info("Distilling student_aux_same...")
    student_aux_same = _distill_students(
        ref_a,
        teacher_same,
        n_batches,
        config,
        device,
        use_all_logits=False,
        name="student_aux_same",
    )
    logger.info("Distilling student_all_same...")
    student_all_same = _distill_students(
        ref_a,
        teacher_same,
        n_batches,
        config,
        device,
        use_all_logits=True,
        name="student_all_same",
    )
    logger.info("Distilling student_aux_diff...")
    student_aux_diff = _distill_students(
        ref_a,
        teacher_diff,
        n_batches,
        config,
        device,
        use_all_logits=False,
        name="student_aux_diff",
    )
    logger.info("Distilling student_all_diff...")
    student_all_diff = _distill_students(
        ref_a,
        teacher_diff,
        n_batches,
        config,
        device,
        use_all_logits=True,
        name="student_all_diff",
    )

    all_models: dict[str, MultiMLP] = {
        "reference": ref_a,
        "teacher": teacher_same,
        "student_aux_same": student_aux_same,
        "student_all_same": student_all_same,
        "student_aux_diff": student_aux_diff,
        "student_all_diff": student_all_diff,
    }
    results: dict[str, list[float]] = {}
    for name, model in all_models.items():
        accs = _evaluate_n(model, test_loader, config.n_classes, device)
        results[name] = accs.tolist()
        logger.info(f"eval {name} | mean acc {accs.mean():.3f} ± {accs.std():.3f}")

    if use_wandb:
        summary: dict[str, float] = {}
        for name, accs in results.items():
            mean = statistics.mean(accs)
            ci95 = 1.96 * statistics.stdev(accs) / math.sqrt(len(accs))
            summary[f"mean/{name}"] = mean
            summary[f"ci95/{name}"] = ci95
        wandb.log(summary)

    return results
