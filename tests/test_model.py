"""Tests for MLP and CNN: output shapes, dtypes, and logit index order."""

import pytest
import torch

from subliminal_learning.config import Config
from subliminal_learning.model import CNN, MLP, MultiMLP


@pytest.fixture
def mlp() -> MLP:
    return MLP(input_shape=(1, 28, 28), n_classes=10, hidden_dims=(64, 64), n_aux=3)


@pytest.fixture
def cnn() -> CNN:
    return CNN(input_channels=1, n_classes=10, n_aux=3)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


def test_mlp_output_shape(mlp: MLP) -> None:
    out = mlp(torch.randn(32, 1, 28, 28))
    assert out.shape == (32, 13)


def test_mlp_class_logit_slice(mlp: MLP) -> None:
    out = mlp(torch.randn(8, 1, 28, 28))
    assert out[:, : mlp.n_classes].shape == (8, 10)
    assert out[:, mlp.n_classes :].shape == (8, 3)


def test_mlp_n_aux_configurable() -> None:
    for n_aux in (1, 3, 5):
        m = MLP(input_shape=(1, 28, 28), n_classes=10, hidden_dims=(32,), n_aux=n_aux)
        assert m(torch.randn(4, 1, 28, 28)).shape == (4, 10 + n_aux)


def test_mlp_input_shape_configurable() -> None:
    """MLP should accept any input shape, flattening it internally."""
    m = MLP(input_shape=(3, 32, 32), n_classes=10, hidden_dims=(64,), n_aux=3)
    assert m(torch.randn(4, 3, 32, 32)).shape == (4, 13)


def test_mlp_n_classes_configurable() -> None:
    m = MLP(input_shape=(1, 28, 28), n_classes=100, hidden_dims=(64,), n_aux=3)
    out = m(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 103)
    assert out[:, :100].shape == (4, 100)
    assert out[:, 100:].shape == (4, 3)


def test_mlp_output_dtype(mlp: MLP) -> None:
    assert mlp(torch.randn(4, 1, 28, 28)).dtype == torch.float32


def test_mlp_batch_independence(mlp: MLP) -> None:
    """Output for one sample must not depend on other samples in the batch."""
    mlp.eval()
    x = torch.randn(4, 1, 28, 28)
    torch.testing.assert_close(mlp(x)[0], mlp(x[:1])[0])


def test_mlp_depth_configurable() -> None:
    for depth in (1, 2, 3):
        m = MLP(input_shape=(32,), n_classes=5, hidden_dims=(16,) * depth, n_aux=2)
        assert m(torch.randn(4, 32)).shape == (4, 7)


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------


def test_cnn_output_shape_mnist(cnn: CNN) -> None:
    out = cnn(torch.randn(32, 1, 28, 28))
    assert out.shape == (32, 13)


def test_cnn_output_shape_cifar() -> None:
    m = CNN(input_channels=3, n_classes=10, n_aux=3)
    out = m(torch.randn(32, 3, 32, 32))
    assert out.shape == (32, 13)


def test_cnn_class_logit_slice(cnn: CNN) -> None:
    out = cnn(torch.randn(8, 1, 28, 28))
    assert out[:, : cnn.n_classes].shape == (8, 10)
    assert out[:, cnn.n_classes :].shape == (8, 3)


def test_cnn_output_dtype(cnn: CNN) -> None:
    assert cnn(torch.randn(4, 1, 28, 28)).dtype == torch.float32


def test_cnn_batch_independence(cnn: CNN) -> None:
    cnn.eval()
    x = torch.randn(4, 1, 28, 28)
    torch.testing.assert_close(cnn(x)[0], cnn(x[:1])[0])


# ---------------------------------------------------------------------------
# MultiMLP
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_mlp() -> MultiMLP:
    return MultiMLP(
        n_models=4, input_shape=(1, 28, 28), n_classes=10, hidden_dims=(64, 64), n_aux=3
    )


def test_multi_mlp_output_shape(multi_mlp: MultiMLP) -> None:
    out = multi_mlp(torch.randn(32, 1, 28, 28))
    assert out.shape == (4, 32, 13)


def test_multi_mlp_model_independence(multi_mlp: MultiMLP) -> None:
    """Each model in the batch should produce different outputs."""
    out = multi_mlp(torch.randn(8, 1, 28, 28))  # [4, 8, 13]
    assert not torch.allclose(out[0], out[1])


def test_from_seeds_matches_individual_mlps() -> None:
    """
    MultiMLP.from_seeds must copy weights faithfully from individually-seeded MLPs.
    """
    config = Config(
        input_shape=(1, 28, 28), n_classes=10, hidden_dims=(64, 64), n_aux=3
    )
    seeds = [0, 1, 2]
    multi = MultiMLP.from_seeds(seeds, config)

    x = torch.randn(8, 1, 28, 28)
    multi_out = multi(x)  # [3, 8, 13]

    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        mlp = MLP.from_config(config)
        torch.testing.assert_close(multi_out[i], mlp(x))
