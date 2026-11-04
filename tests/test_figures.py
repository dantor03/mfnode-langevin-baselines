"""The figures are drawn from arrays alone: no torch, no training.

``figure_data.make`` builds synthetic arrays with the keys and shapes that each
experiment saves; a static check below keeps those keys in step with the
``publish(...)`` call of every experiment.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pytest
from figure_data import make

import mfnode.plotting as mp
from mfnode.plotting.style import eps_colors, prior_density_1d

ROOT = Path(__file__).resolve().parents[1]
NAMES = list("ABCDEFGHIJK")
OUTPUTS = {"A": ["A_feature_evolution.png"],
           "B": ["B1_convergence_curves.png", "B2_decision_boundaries.png"],
           "C": ["C_pl_verification.png"], "D": ["D_genericity.png"],
           "E": ["E_parameter_robustness.png"],
           "F": ["F_circles_parameter_distribution.png"],
           "G": ["G_convergence_problem.png"], "H": ["H_psgld_diagnostic.png"],
           "I": ["I_regularizer_comparison.png"], "J": ["J_california_regression.png"],
           "K": ["K_epsilon_sweep.png"]}
PROTOCOLS = ("d1", "d2", "d3")


@pytest.mark.parametrize("name", NAMES)
def test_figure_is_drawn_from_synthetic_data(name, tmp_path):
    outs = mp.draw(name, make(name), out_dir=tmp_path)
    assert [o.name for o in outs] == OUTPUTS[name]
    for o in outs:
        assert o.stat().st_size > 20_000          # a real image, not an empty canvas


def test_save_load_round_trip(tmp_path):
    data = make("H")
    path = mp.save_data("H", data, data_dir=tmp_path)
    back = mp.load_data("H", data_dir=tmp_path)
    assert path.name == "H.npz" and set(back) == set(data)
    for k in data:
        np.testing.assert_allclose(back[k], np.asarray(data[k]))


def _expand(keys):
    out = set()
    for k in keys:
        out |= {k.replace("{p}", p) for p in PROTOCOLS} if "{p}" in k else {k}
    return out


def _plot_keys(name):
    src = (ROOT / "mfnode/plotting/figures.py").read_text()
    body = src[src.index(f"def plot_{name}("):]
    body = body[:body.index("\ndef plot_", 1)] if "\ndef plot_" in body[1:] else body
    keys = re.findall(r"""d\[f?["']([^"']+)["']\]""", body)
    return _expand(keys)


def _saved_keys(name):
    """Keyword names of the dict passed to publish('<name>', ...), plus D's f-string keys."""
    tree = ast.parse((ROOT / f"mfnode/experiments/exp_{name.lower()}.py").read_text())
    keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "dict"):
            keys |= {kw.arg for kw in node.keywords if kw.arg}
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
                elif isinstance(k, ast.JoinedStr):
                    keys.add("".join(v.value if isinstance(v, ast.Constant) else "{p}"
                                     for v in k.values))
    return _expand(keys)


@pytest.mark.parametrize("name", NAMES)
def test_experiment_saves_every_key_its_figure_reads(name):
    missing = _plot_keys(name) - _saved_keys(name)
    assert not missing, f"exp_{name.lower()}.py does not save {sorted(missing)}"


@pytest.mark.parametrize("name", NAMES)
def test_synthetic_data_has_every_key_the_experiment_saves(name):
    missing = (_saved_keys(name) & _plot_keys(name)) - set(make(name))
    assert not missing


def test_prior_marginal_is_normalized():
    g = np.linspace(-8, 8, 20001)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    assert trapezoid(prior_density_1d(g), g) == pytest.approx(1.0, rel=1e-6)


def test_eps_colors_are_ordered_and_distinct():
    cols = eps_colors([0, 0.001, 0.01, 0.1, 0.5])
    assert len(set(cols)) == 5
