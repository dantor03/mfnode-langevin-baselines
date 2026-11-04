import math

import numpy as np
import pytest
import torch

from mfnode.metrics import (
    mmd_loss_train, mmd_rbf, sample_prior_langevin, sinkhorn_loss, wasserstein_1d,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def gauss(n, d=5, shift=0.0):
    return torch.randn(n, d) + shift


# ── MMD² ─────────────────────────────────────────────────────────────────────

def test_mmd_of_two_samples_of_the_same_law_is_near_zero():
    assert abs(mmd_rbf(gauss(300), gauss(300))) < 0.01


def test_mmd_detects_a_shift_and_grows_with_it():
    X = gauss(300)
    vals = [mmd_rbf(X, gauss(300, shift=s)) for s in (0.0, 0.5, 1.0, 2.0)]
    assert vals[1] > 5 * abs(vals[0])
    assert vals == sorted(vals)


def test_training_mmd_vanishes_on_identical_samples():
    X = gauss(64)
    assert float(mmd_loss_train(X, X, sigma=1.0)) == pytest.approx(0.0, abs=1e-6)


def test_training_mmd_gradient_moves_particles_towards_the_prior():
    Y = gauss(256)
    X = gauss(64, shift=1.5).requires_grad_(True)
    loss = mmd_loss_train(X, Y, sigma=2.0)
    loss.backward()
    with torch.no_grad():
        X_new = X - 5.0 * X.grad
    assert X.grad.abs().sum() > 0
    assert float(mmd_loss_train(X_new, Y, sigma=2.0)) < float(loss)


# ── Sinkhorn divergence ──────────────────────────────────────────────────────

def test_sinkhorn_divergence_vanishes_on_identical_samples():
    X = gauss(64)
    assert float(sinkhorn_loss(X, X)) == pytest.approx(0.0, abs=1e-5)


def test_sinkhorn_divergence_is_positive_and_grows_with_the_shift():
    X = gauss(64)
    vals = [float(sinkhorn_loss(X, gauss(64, shift=s))) for s in (0.5, 1.0, 2.0)]
    assert vals[0] > 0
    assert vals == sorted(vals)


def test_sinkhorn_gradient_flows_to_the_particles():
    X = gauss(32, shift=1.0).requires_grad_(True)
    sinkhorn_loss(X, gauss(64)).backward()
    assert torch.isfinite(X.grad).all() and X.grad.abs().sum() > 0


# ── W₁ and the ULA prior sampler ─────────────────────────────────────────────

def test_wasserstein_1d_of_a_translation():
    x = np.random.default_rng(0).normal(size=500)
    assert wasserstein_1d(x, x + 0.7) == pytest.approx(0.7)


def test_ula_samples_match_the_prior_moments():
    # ν^∞ ∝ exp(−ℓ(a)), ℓ(a) = 0.05 Σ a⁴ + 0.5 Σ a²: independent coordinates.
    a = np.linspace(-8, 8, 20001)
    w = np.exp(-(0.05 * a**4 + 0.5 * a**2))
    var_exact = float((a**2 * w).sum() / w.sum())
    S = sample_prior_langevin(2000, dim=5).numpy()
    assert S.shape == (2000, 5)
    assert abs(S.mean()) < 0.1
    assert S.var(axis=0).mean() == pytest.approx(var_exact, rel=0.1)
    assert math.isfinite(var_exact) and var_exact < 1.0   # lighter tails than N(0, 1)
