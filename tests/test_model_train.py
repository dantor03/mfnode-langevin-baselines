import math

import pytest
import torch

from mfnode.data import get_moons
from mfnode.metrics import extract_particles, particles_live, sample_prior_langevin
from mfnode.model import MeanFieldResNet
from mfnode.train import mu_pl_estimate, train


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope="module")
def moons():
    X, y, _, _ = get_moons(n=120)
    return X.cpu(), y.cpu()


def test_forward_and_particles_shapes():
    model = MeanFieldResNet(M=16)
    x = torch.randn(10, 2)
    assert model(x).shape == (10,)
    assert extract_particles(model).shape == (16, 5)
    assert particles_live(model).requires_grad


def test_warm_start_round_trip():
    neurons = torch.randn(8, 5)
    model = MeanFieldResNet(M=8, warm_start_neurons=neurons)
    assert torch.allclose(extract_particles(model), neurons)


def test_modes_are_exclusive(moons):
    X, y = moons
    with pytest.raises(ValueError):
        train(MeanFieldResNet(M=8), X, y, 0.01, n_epochs=1, use_sgld=True, use_sgd=True,
              verbose=False)
    with pytest.raises(ValueError):
        train(MeanFieldResNet(M=8), X, y, 0.01, n_epochs=1, use_mmd=True, verbose=False)


@pytest.mark.parametrize("mode", ["adam", "sgld", "mmd", "sinkhorn"])
def test_short_training_run(mode, moons):
    X, y = moons
    kw = {"sgld": dict(use_sgld=True), "mmd": dict(use_mmd=True),
          "sinkhorn": dict(use_sinkhorn=True), "adam": {}}[mode]
    if mode in ("mmd", "sinkhorn"):
        kw["prior_samples"] = sample_prior_langevin(200, n_burnin=500)
    hist = train(MeanFieldResNet(M=16), X, y, 0.01, n_epochs=60, verbose=False, **kw)
    assert len(hist["loss"]) == 60
    assert all(math.isfinite(v) for v in hist["loss"])
    assert hist["loss"][-1] < hist["loss"][0]
    assert hist["J_star"] == min(hist["loss"])
    assert mu_pl_estimate(hist) >= 0.0
