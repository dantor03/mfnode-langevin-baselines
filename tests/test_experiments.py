"""Smoke test of every experiment with tiny settings: it runs, saves its data and its
figure, and the figure can be redrawn from the saved data alone.

The numbers are meaningless at this size (2–3 epochs); only the plumbing is tested.
Figures go to the temporary MFNODE_OUTPUT_DIR set in conftest.py.
"""

from pathlib import Path

import pytest
import torch

import mfnode.plotting as mp
from mfnode.experiments import (
    experiment_A, experiment_B, experiment_C, experiment_D, experiment_E_robustness,
    experiment_F, experiment_G, experiment_H, experiment_I, experiment_J, experiment_K,
)
from mfnode.paths import OUTPUT_DIR
from test_figures import OUTPUTS

EPS = [0.0, 0.01]
RUNS = {
    "A": lambda: experiment_A(n_epochs=3),
    "B": lambda: experiment_B(epsilons=EPS, n_epochs=3),
    "C": lambda: experiment_C(epsilons=EPS, n_epochs=3),
    "D": lambda: experiment_D(n_seeds=2, n_epochs=2),
    "E": lambda: experiment_E_robustness(n_seeds=2, n_epochs=2),
    "F": lambda: experiment_F(n_seeds=2, n_epochs=2),
    "G": lambda: experiment_G(n_seeds=1, n_epochs=2),
    "H": lambda: experiment_H(n_epochs=3, n_snapshot_steps=20, thin=10, n_prior_samples=60),
    "I": lambda: experiment_I(n_epochs=3, n_prior=60, n_snapshot_steps=20, thin=10),
    "J": lambda: experiment_J(n_epochs=3, n_prior=60, n_train=80, n_test=40,
                              n_snapshot_steps=20, thin=10),
    "K": lambda: experiment_K(epsilon_grid=[0.01, 0.1], n_epochs=2, n_prior=60,
                              n_train=80, n_test=40),
}


@pytest.fixture(scope="module")
def california():
    try:
        from sklearn.datasets import fetch_california_housing
        fetch_california_housing()
    except Exception as exc:                      # no network and no local copy
        pytest.skip(f"California Housing not available: {exc}")


@pytest.mark.parametrize("name", list(RUNS))
def test_experiment_saves_data_and_figure(name, request):
    if name in "JK":
        request.getfixturevalue("california")
    torch.manual_seed(0)
    RUNS[name]()
    assert mp.data_path(name).exists()
    for fname in OUTPUTS[name]:
        assert (Path(OUTPUT_DIR) / fname).stat().st_size > 20_000
    # The figure is reproducible from the saved file alone.
    redrawn = mp.draw(name, out_dir=Path(OUTPUT_DIR) / "redraw")
    assert [p.name for p in redrawn] == OUTPUTS[name]
