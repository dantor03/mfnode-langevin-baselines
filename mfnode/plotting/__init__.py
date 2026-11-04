"""Figures of the experiments, drawn from saved arrays (no torch needed).

Each experiment computes its results, stores them with :func:`save_data` in
``figures/data/<X>.npz`` and draws them with :func:`draw`.  The figures can be
redrawn later from those files alone::

    python -m mfnode --plot-only            # every experiment with saved data
    python -m mfnode -e H --plot-only       # only H
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..paths import DATA_DIR, OUTPUT_DIR
from .figures import PLOTS

__all__ = ['PLOTS', 'data_path', 'save_data', 'load_data', 'draw', 'publish', 'redraw']


def data_path(name: str, data_dir=None) -> Path:
    return Path(data_dir or DATA_DIR) / f'{name}.npz'


def _to_numpy(v):
    if hasattr(v, 'detach'):                     # torch tensor, without importing torch
        v = v.detach().cpu().numpy()
    return np.asarray(v)


def save_data(name: str, data: dict, data_dir=None) -> Path:
    """Write the arrays of one experiment to ``<data_dir>/<name>.npz``."""
    path = data_path(name, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: _to_numpy(v) for k, v in data.items()})
    return path


def load_data(name: str, data_dir=None) -> dict:
    with np.load(data_path(name, data_dir), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def draw(name: str, data: dict | None = None, out_dir=None) -> list[Path]:
    """Draw the figure(s) of experiment ``name`` from ``data`` or its saved file."""
    if data is None:
        data = load_data(name)
    data = {k: _to_numpy(v) for k, v in data.items()}
    return PLOTS[name](data, Path(out_dir or OUTPUT_DIR))


def publish(name: str, data: dict) -> list[Path]:
    """Save the data of experiment ``name`` and draw its figure(s)."""
    path = save_data(name, data)
    print(f"  → datos: {path}")
    outs = draw(name, data)
    for out in outs:
        print(f"  → {out}")
    return outs


def redraw(names=None) -> dict[str, list[Path]]:
    """Redraw every experiment in ``names`` (default: all) that has saved data."""
    done = {}
    for name in names or PLOTS:
        if data_path(name).exists():
            done[name] = draw(name)
    return done
