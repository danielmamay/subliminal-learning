"""MLP and CNN with auxiliary logits for subliminal learning experiments."""

import math

import torch
import torch.nn as nn
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor

from .config import Config


class Model(nn.Module):
    """Base class for models used in subliminal learning experiments.

    Subclasses expose ``n_classes`` and ``n_aux`` and implement ``from_config``.
    Output logits are ordered as ``[class_logits | aux_logits]``:

    - ``logits[:, :n_classes]`` — trained via cross-entropy (teacher) or left
      untrained (student aux-only condition); used for accuracy evaluation.
    - ``logits[:, n_classes:]`` — used as KL-divergence targets during distillation.
    """

    n_classes: int
    n_aux: int

    @classmethod
    def from_config(cls, config: "Config") -> "Model":
        raise NotImplementedError


class MLP(Model):
    """Feedforward MLP.

    Accepts image tensors of any spatial shape; flattens internally via
    ``nn.Flatten``, so it is compatible with the same ``(batch, C, H, W)``
    loaders used by :class:`CNN`.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        n_classes: int,
        hidden_dims: tuple[int, ...],
        n_aux: int,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_aux = n_aux

        input_dim = math.prod(input_shape)
        dims = (input_dim,) + hidden_dims + (n_classes + n_aux,)
        layers: list[nn.Module] = [nn.Flatten()]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, config: "Config") -> "MLP":
        return cls(
            config.input_shape, config.n_classes, config.hidden_dims, config.n_aux
        )

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Tensor) -> Float[Tensor, "batch n_out"]:
        return self.net(x)  # type: ignore[return-value]


class CNN(Model):
    """Small convolutional network.

    Architecture: Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) →
    AdaptiveAvgPool(4x4) → Flatten → Linear(n_classes + n_aux).

    ``AdaptiveAvgPool2d`` decouples the classifier head from input spatial
    resolution, so the same architecture works for 28x28 (MNIST / Fashion-MNIST)
    and 32x32 (CIFAR-10) inputs.
    """

    def __init__(self, input_channels: int, n_classes: int, n_aux: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_aux = n_aux
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, n_classes + n_aux),
        )

    @classmethod
    def from_config(cls, config: "Config") -> "CNN":
        return cls(config.input_shape[0], config.n_classes, config.n_aux)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch n_out"]:
        return self.classifier(self.features(x))  # type: ignore[return-value]


class MultiLinear(nn.Module):
    """N independent linear layers applied simultaneously.

    Args:
        n_models: Number of parallel models.
        d_in: Input features per model.
        d_out: Output features per model.

    Shape:
        - Input: ``(n_models, batch, d_in)``
        - Output: ``(n_models, batch, d_out)``
    """

    def __init__(self, n_models: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1.0 / math.sqrt(d_in))

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[Tensor, "n_models batch d_in"]
    ) -> Float[Tensor, "n_models batch d_out"]:
        return x.matmul(self.weight.transpose(1, 2)) + self.bias[:, None, :]


class MultiMLP(nn.Module):
    """N independent MLPs applied simultaneously via batched matmul.

    Vectorised analogue of :class:`MLP`: same architecture, but weights are
    stored as ``[n_models, d_out, d_in]`` tensors so all N models execute in a
    single batched matmul per layer, with a standard :class:`torch.optim.Optimizer`.

    Use :meth:`from_seeds` to create with independently-seeded initialisations.
    """

    def __init__(
        self,
        n_models: int,
        input_shape: tuple[int, ...],
        n_classes: int,
        hidden_dims: tuple[int, ...],
        n_aux: int,
    ) -> None:
        super().__init__()
        self.n_models = n_models
        self.n_classes = n_classes
        self.n_aux = n_aux

        input_dim = math.prod(input_shape)
        dims = (input_dim,) + hidden_dims + (n_classes + n_aux,)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(MultiLinear(n_models, dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Tensor) -> Float[Tensor, "n_models batch n_out"]:
        flat = x.flatten(1)
        expanded = flat.unsqueeze(0).expand(self.n_models, -1, -1)
        return self.net(expanded)  # type: ignore[return-value]

    @classmethod
    def from_seeds(cls, seeds: list[int], config: "Config") -> "MultiMLP":
        """Create a :class:`MultiMLP` with N independently-seeded initialisations.

        Instantiates N individual :class:`MLP` models (one per seed), then stacks
        their weights into the corresponding :class:`MultiLinear` layers.
        """
        individual_mlps: list[MLP] = []
        for seed in seeds:
            torch.manual_seed(seed)
            individual_mlps.append(MLP.from_config(config))

        multi = cls(
            n_models=len(seeds),
            input_shape=config.input_shape,
            n_classes=config.n_classes,
            hidden_dims=config.hidden_dims,
            n_aux=config.n_aux,
        )

        src_layers = [
            [m for m in mlp.net if isinstance(m, nn.Linear)] for mlp in individual_mlps
        ]
        dst_layers = [m for m in multi.net if isinstance(m, MultiLinear)]
        for dst, *srcs in zip(dst_layers, *src_layers):
            dst.weight.data = torch.stack([s.weight.data for s in srcs])
            dst.bias.data = torch.stack([s.bias.data for s in srcs])

        return multi


#: Maps architecture name → model class.
MODEL_REGISTRY: dict[str, type[Model]] = {
    "mlp": MLP,
    "cnn": CNN,
}
