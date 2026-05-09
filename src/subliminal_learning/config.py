"""Experiment configuration."""

import math
from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters for a subliminal learning experiment.

    Designed to be dataset- and architecture-agnostic: set ``input_shape``,
    ``n_classes``, and ``hidden_dims`` to adapt to any classification dataset.
    """

    # Dataset geometry
    input_shape: tuple[int, ...] = (1, 28, 28)  # (C, H, W)
    n_classes: int = 10

    @property
    def input_dim(self) -> int:
        """Flattened input dimension; useful for MLP and noise generation."""
        return math.prod(self.input_shape)

    # Model architecture
    hidden_dims: tuple[int, ...] = (256, 256)
    n_aux: int = 3

    # Training
    n_epochs_teacher: int = 5
    n_epochs_student: int = 5
    batch_size: int = 256
    lr: float = 1e-3

    # Distillation temperature (T=1 matches the paper)
    temperature: float = 1.0

    # Experiment
    n_seeds: int = 100
