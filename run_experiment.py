#!/usr/bin/env python
"""CLI entry point for the subliminal learning experiment."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
from loguru import logger

from subliminal_learning.config import Config
from subliminal_learning.data import DATASETS, get_loaders
from subliminal_learning.experiment_cnn import run_experiment_cnn
from subliminal_learning.experiment_mlp import run_experiment_mlp
from subliminal_learning.model import MLP, MODEL_REGISTRY
from subliminal_learning.plot import plot_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Subliminal learning experiment")
    parser.add_argument("--dataset", choices=list(DATASETS), default="mnist")
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--n-aux", type=int, default=3)
    parser.add_argument("--n-epochs-teacher", type=int, default=5)
    parser.add_argument("--n-epochs-student", type=int, default=5)
    parser.add_argument("--arch", choices=list(MODEL_REGISTRY), default="mlp")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-epoch loss/accuracy (DEBUG level)",
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="subliminal-learning-vision")
    parser.add_argument(
        "--output",
        default=None,
        help="Stem or full path for outputs (default: auto-derived from config)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")
    if not args.verbose:
        logger.info("Pass --verbose to see per-epoch loss/accuracy during training.")

    ds = DATASETS[args.dataset]
    config = Config(
        input_shape=ds.input_shape,
        n_classes=ds.n_classes,
        n_seeds=args.n_seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=tuple(args.hidden_dims),
        n_aux=args.n_aux,
        n_epochs_teacher=args.n_epochs_teacher,
        n_epochs_student=args.n_epochs_student,
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.output:
        stem = Path(args.output).with_suffix("")
    else:
        stem = results_dir / (
            f"{args.dataset}_{args.arch}"
            f"_n{config.n_seeds}"
            f"_t{config.n_epochs_teacher}"
            f"_s{config.n_epochs_student}"
            f"_a{config.n_aux}"
        )

    plot_path = stem.with_suffix(".png")
    config_path = stem.with_suffix(".json")

    device = torch.device(args.device)
    train_loader, test_loader = get_loaders(args.dataset, config.batch_size, device)

    model_cls = MODEL_REGISTRY[args.arch]
    runner = run_experiment_mlp if model_cls is MLP else run_experiment_cnn
    results = runner(
        config,
        device,
        train_loader,
        test_loader,
        model_cls,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    title = f"Subliminal learning: {args.dataset} {args.arch.upper()} experiment"
    plot_results(
        results, n_classes=config.n_classes, output_path=str(plot_path), title=title
    )

    config_data = {
        **dataclasses.asdict(config),
        "dataset": args.dataset,
        "arch": args.arch,
    }
    config_path.write_text(json.dumps(config_data, indent=2))

    if args.wandb:
        import wandb

        wandb.log({"results": wandb.Image(str(plot_path))})
        wandb.save(str(config_path))
        wandb.finish()

    print(f"Results saved to {plot_path} (config: {config_path})")


if __name__ == "__main__":
    main()
