"""CNN experiment: sequential per-seed loop for non-MLP architectures."""

import copy
import dataclasses
import math
import statistics

import torch
import torch.nn.functional as F
from loguru import logger

from .config import Config
from .data import LabeledLoader, sample_noise
from .model import Model


def _train_teacher(
    model: Model,
    train_loader: LabeledLoader,
    config: Config,
    device: torch.device,
    name: str = "teacher",
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    for epoch in range(config.n_epochs_teacher):
        epoch_loss = epoch_acc = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits[:, : config.n_classes], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += (
                (logits[:, : config.n_classes].argmax(-1) == y).float().mean().item()
            )
        n = len(train_loader)
        logger.debug(
            f"{name} | epoch {epoch + 1}/{config.n_epochs_teacher} "
            f"| loss {epoch_loss / n:.4f} | acc {epoch_acc / n:.3f}"
        )


def _distill_student(
    student: Model,
    teacher: Model,
    n_batches_per_epoch: int,
    config: Config,
    device: torch.device,
    *,
    use_all_logits: bool,
    name: str = "student",
) -> None:
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr)
    teacher.eval()
    student.train()
    T = config.temperature
    for epoch in range(config.n_epochs_student):
        epoch_loss = 0.0
        for _ in range(n_batches_per_epoch):
            x = sample_noise(config.batch_size, config.input_shape, device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            t_slice = t_logits if use_all_logits else t_logits[:, config.n_classes :]
            s_slice = s_logits if use_all_logits else s_logits[:, config.n_classes :]
            loss = F.kl_div(
                F.log_softmax(s_slice / T, dim=-1),
                F.softmax(t_slice / T, dim=-1),
                reduction="batchmean",
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.debug(
            f"{name} | epoch {epoch + 1}/{config.n_epochs_student}"
            f" | loss {epoch_loss / n_batches_per_epoch:.4f}"
        )


def _accuracy(model: Model, loader: LabeledLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)[:, : model.n_classes].argmax(-1)
            correct += int((preds == y).sum())
            total += y.size(0)
    return correct / total


def _run_trial(
    seed: int,
    config: Config,
    device: torch.device,
    train_loader: LabeledLoader,
    test_loader: LabeledLoader,
    model_cls: type[Model],
) -> dict[str, float]:
    """Train two teachers and four students for a single seed pair.

    Uses two reference initialisations:

    - **ref_A** (``seed``): starting point for the same-init teacher and all students.
    - **ref_B** (``seed + 1_000_000``): starting point for the different-init teacher.
    """
    n_batches = len(train_loader)

    torch.manual_seed(seed)
    ref_a = model_cls.from_config(config).to(device)
    torch.manual_seed(seed + 1_000_000)
    ref_b = model_cls.from_config(config).to(device)

    teacher_same = copy.deepcopy(ref_a)
    _train_teacher(
        teacher_same, train_loader, config, device, name="teacher (same init)"
    )

    teacher_diff = copy.deepcopy(ref_b)
    _train_teacher(
        teacher_diff, train_loader, config, device, name="teacher (diff init)"
    )

    student_aux_same = copy.deepcopy(ref_a)
    _distill_student(
        student_aux_same,
        teacher_same,
        n_batches,
        config,
        device,
        use_all_logits=False,
        name="student_aux_same",
    )

    student_all_same = copy.deepcopy(ref_a)
    _distill_student(
        student_all_same,
        teacher_same,
        n_batches,
        config,
        device,
        use_all_logits=True,
        name="student_all_same",
    )

    student_aux_diff = copy.deepcopy(ref_a)
    _distill_student(
        student_aux_diff,
        teacher_diff,
        n_batches,
        config,
        device,
        use_all_logits=False,
        name="student_aux_diff",
    )

    student_all_diff = copy.deepcopy(ref_a)
    _distill_student(
        student_all_diff,
        teacher_diff,
        n_batches,
        config,
        device,
        use_all_logits=True,
        name="student_all_diff",
    )

    return {
        "reference": _accuracy(ref_a, test_loader, device),
        "teacher": _accuracy(teacher_same, test_loader, device),
        "student_aux_same": _accuracy(student_aux_same, test_loader, device),
        "student_all_same": _accuracy(student_all_same, test_loader, device),
        "student_aux_diff": _accuracy(student_aux_diff, test_loader, device),
        "student_all_diff": _accuracy(student_all_diff, test_loader, device),
    }


def run_experiment_cnn(
    config: Config,
    device: torch.device,
    train_loader: LabeledLoader,
    test_loader: LabeledLoader,
    model_cls: type[Model],
    *,
    use_wandb: bool = True,
    wandb_project: str = "subliminal-learning-vision",
) -> dict[str, list[float]]:
    """Run the full experiment across ``config.n_seeds`` independent trials.

    Returns:
        Mapping from condition name to list of per-trial accuracies
        (length ``n_seeds``).
    """
    import wandb

    if use_wandb:
        wandb.init(project=wandb_project, config=dataclasses.asdict(config))

    results: dict[str, list[float]] = {}

    for seed in range(config.n_seeds):
        logger.info(f"Trial {seed + 1}/{config.n_seeds}")
        trial = _run_trial(seed, config, device, train_loader, test_loader, model_cls)
        for name, acc in trial.items():
            results.setdefault(name, []).append(acc)
        if use_wandb:
            wandb.log({"seed": seed, **{f"acc/{k}": v for k, v in trial.items()}})

    if use_wandb:
        summary: dict[str, float] = {}
        for name, accs in results.items():
            mean = statistics.mean(accs)
            ci95 = 1.96 * statistics.stdev(accs) / math.sqrt(len(accs))
            summary[f"mean/{name}"] = mean
            summary[f"ci95/{name}"] = ci95
        wandb.log(summary)

    return results
